"""
De novo motif and positional feature discovery for translation regulation.

All position metrics are CDS-start-aligned (pos 0 = first nt of start codon),
making them comparable across variable-length sequences.

Strategy:
  Phase 1 — model-intrinsic (fast):
    1A. Attention positional importance (single forward per sample)
    1B. Input saliency (single forward+backward per sample)
    1C. AdaLN gene attribution (weight inspection, zero cost)
  Phase 2 — targeted mutagenesis (cost-aware):
    Only mutate top-K hotspot positions from Phase 1.
"""

import os, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import defaultdict, Counter
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# Helper: unwrap DDP model
# ============================================================
def _unwrap(model):
    """Unwrap DistributedDataParallel to get the raw model."""
    return model.module if hasattr(model, 'module') else model


# ============================================================
# Helper: extract a CDS-aligned sample from dataset
# ============================================================
def _extract_sample(dataset, idx):
    """
    Extract sample and compute CDS-aligned metadata.
    All CDS coordinates 0-based.
    """
    uuid, species, ct, ev, mi, se, ce = dataset[idx]

    se_np = se.cpu().numpy() if torch.is_tensor(se) else np.array(se)
    ce_np = ce.cpu().numpy() if torch.is_tensor(ce) else np.array(ce)

    # Expression vector: keep as-is
    # dataset may return empty array if cell_type not in cell_expr_dict
    if torch.is_tensor(ev):
        ev_np = ev.cpu().numpy()
    else:
        ev_np = np.array(ev) if ev is not None else None
    # Normalize: shape-0 array -> None (will use fallback later)
    if ev_np is not None and ev_np.ndim == 1 and ev_np.shape[0] == 0:
        ev_np = None

    cds_start = int(mi.get('cds_start_pos', -1)) - 1 if isinstance(mi, dict) else -1
    cds_end = int(mi.get('cds_end_pos', -1)) - 1 if isinstance(mi, dict) else -1

    tid = str(uuid).rsplit('-', 2)[0] if '-' in str(uuid) else str(uuid).split('.')[0]

    return {
        'se': se_np, 'ce': ce_np, 'ev': ev_np,
        'cds_start_0': cds_start, 'cds_end_0': cds_end,
        'L': se_np.shape[0], 'ct': ct, 'species': species, 'tid': tid,
        'valid': cds_start >= 0 and cds_end > cds_start
    }


# ============================================================
# Phase 1A: Attention positional importance
# ============================================================
def extract_attention_positional_importance(model, dataset, n_samples=200,
                                             max_len=1200, device=None):
    """
    Forward samples, extract per-layer attention received per position,
    aggregate aligned to CDS start.

    Args:
        model: trained model (can be DDP-wrapped)
        dataset: TranslationDataset
        device: torch device (auto-detected if None)

    Returns:
        DataFrame: layer, pos_from_cds_start, mean_attn, std_attn, n_contrib
    """
    raw = _unwrap(model)
    if device is None:
        device = next(raw.parameters()).device
    raw.eval()

    n_layers = len(raw.encoder.encoder_layers)
    n_heads = raw.n_heads
    head_dim = raw.encoder.encoder_layers[0].multi_headed_attention.head_dim

    accum = defaultdict(lambda: {'sum': 0.0, 'sum_sq': 0.0, 'n': 0})
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    valid_count = 0

    for idx in tqdm(indices, desc="Attention positional importance"):
        s = _extract_sample(dataset, idx)
        if not s['valid'] or s['L'] > max_len:
            continue
        se = torch.from_numpy(s['se']).float().unsqueeze(0).to(device)
        ce = torch.from_numpy(s['ce']).float().unsqueeze(0).to(device)
        # Use model's mean_expr_vector as fallback if ev is missing
        if s['ev'] is not None and len(s['ev']) > 0:
            ev = torch.from_numpy(s['ev']).float().unsqueeze(0).to(device)
        else:
            ev = None
        L = s['L']
        cds_start = s['cds_start_0']

        with torch.no_grad():
            # Resolve expression + species through model's internal pipeline
            # Pass ev=None to use fallback (mean_expr_vector)
            resolved_expr = raw._resolve_expr_vector(
                cell_type=s['ct'], expr_vector=ev, batch_size=1
            ).to(device)
            species_idx = raw._normalize_species(s['species'], 1).to(device)
            species_emb = raw.species_embedding(species_idx)
            combined_env = torch.cat([resolved_expr, species_emb], dim=-1)
            compact_style = raw.expr_projector(combined_env)

            src_embs = raw.src_emb(se, ce)
            src_mask = (se[:, :, 0] != 0).to(device)
            src_reps = src_embs

            for layer_idx, enc_layer in enumerate(raw.encoder.encoder_layers):
                # --- Attention sublayer ---
                sub = enc_layer.sublayers[0]
                style = sub.adaLN_modulation(compact_style)
                gamma, beta, alpha = style.chunk(3, dim=-1)
                normed = (1 + gamma.unsqueeze(1)) * sub.LN(src_reps) + beta.unsqueeze(1)

                attn_mod = enc_layer.multi_headed_attention
                bs_, Lc, d = normed.shape

                q = attn_mod.toqueries(normed).view(bs_, Lc, n_heads, head_dim).transpose(1, 2)
                k = attn_mod.tokeys(normed).view(bs_, Lc, n_heads, head_dim).transpose(1, 2)
                v = attn_mod.tovalues(normed).view(bs_, Lc, n_heads, head_dim).transpose(1, 2)

                if hasattr(attn_mod, 'RoPE'):
                    q = attn_mod.RoPE(q)
                    k = attn_mod.RoPE(k)

                scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(head_dim)
                mask = src_mask[:, :Lc].unsqueeze(1).unsqueeze(2)
                scores.masked_fill_(~mask, float('-inf'))
                attn_w = torch.softmax(scores, dim=-1)

                # Column sum: attention received per position
                received = attn_w.sum(dim=2).mean(dim=1)[0].cpu().numpy()

                for pos in range(Lc):
                    rel_pos = pos - cds_start
                    accum[(layer_idx, rel_pos)]['sum'] += float(received[pos])
                    accum[(layer_idx, rel_pos)]['sum_sq'] += float(received[pos]) ** 2
                    accum[(layer_idx, rel_pos)]['n'] += 1

                # Complete sublayer for next layers
                attn_out = torch.matmul(attn_w, v)
                attn_out = attn_out.transpose(1, 2).reshape(bs_, Lc, n_heads * head_dim)
                attn_out = attn_mod.unifyheads(attn_out)
                if hasattr(attn_mod, 'dropout'):
                    attn_out = attn_mod.dropout(attn_out)
                src_reps = src_reps + alpha.unsqueeze(1) * sub.dropout(attn_out)

                # --- FFN sublayer ---
                sub2 = enc_layer.sublayers[1]
                style2 = sub2.adaLN_modulation(compact_style)
                gamma2, beta2, alpha2 = style2.chunk(3, dim=-1)
                normed2 = (1 + gamma2.unsqueeze(1)) * sub2.LN(src_reps) + beta2.unsqueeze(1)
                ffn_out = sub2.dropout(enc_layer.ffn(normed2))
                src_reps = src_reps + alpha2.unsqueeze(1) * ffn_out

        valid_count += 1

    records = []
    for (layer, pos), v in accum.items():
        if v['n'] >= 5:
            mean = v['sum'] / v['n']
            std = np.sqrt(max(0, v['sum_sq'] / v['n'] - mean ** 2))
            records.append({
                'layer': layer, 'pos_from_cds_start': pos,
                'mean_attn': mean, 'std_attn': std, 'n_contrib': v['n'],
            })

    df = pd.DataFrame(records)
    print(f"Attention aggregated: {len(df)} entries from {valid_count} samples")
    return df


# ============================================================
# Phase 1B: Input saliency
# ============================================================
def compute_saliency_profile(model, dataset, n_samples=100, max_len=1200,
                              device=None):
    """
    Compute d(TE)/d(one_hot), aggregated by CDS-start-aligned position.
    """
    raw = _unwrap(model)
    if device is None:
        device = next(raw.parameters()).device
    raw.eval()

    accum = defaultdict(lambda: {'sum': 0.0, 'sum_sq': 0.0, 'n': 0})
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    valid_count = 0

    for idx in tqdm(indices, desc="Input saliency"):
        s = _extract_sample(dataset, idx)
        if not s['valid'] or s['L'] > max_len:
            continue
        se = torch.from_numpy(s['se']).float().unsqueeze(0).to(device).requires_grad_(True)
        ce = torch.from_numpy(s['ce']).float().unsqueeze(0).to(device)
        if s['ev'] is not None and len(s['ev']) > 0:
            ev = torch.from_numpy(s['ev']).float().unsqueeze(0).to(device)
        else:
            ev = None
        L, cds_start, cds_end = s['L'], s['cds_start_0'], s['cds_end_0']

        # Use raw.forward directly to preserve gradients
        # (model.predict wraps with torch.no_grad, which kills gradients)
        raw.eval()
        with torch.enable_grad():
            # Resolve expression and species
            resolved_expr = raw._resolve_expr_vector(
                cell_type=s['ct'], expr_vector=ev, batch_size=1
            ).to(device)
            species_idx = raw._normalize_species(s['species'], 1).to(device)

            out = raw.forward(
                seq_batch=se, count_batch=ce,
                expr_vector=resolved_expr, species=species_idx,
                head_names=['count'],
            )
            pred = out['count']
            if isinstance(pred, dict):
                pred = pred.get('profile', pred)
            te = pred[0, cds_start:cds_end:3, 0].mean()

        te.backward()
        grad = se.grad[0].detach().cpu().numpy()
        sal = np.abs(grad).sum(axis=-1)

        for pos in range(L):
            rel_pos = pos - cds_start
            accum[rel_pos]['sum'] += float(sal[pos])
            accum[rel_pos]['sum_sq'] += float(sal[pos]) ** 2
            accum[rel_pos]['n'] += 1

        se.grad = None
        valid_count += 1

    records = []
    for pos, v in accum.items():
        if v['n'] >= 5:
            mean = v['sum'] / v['n']
            std = np.sqrt(max(0, v['sum_sq'] / v['n'] - mean ** 2))
            records.append({
                'pos_from_cds_start': pos,
                'mean_saliency': mean, 'std_saliency': std, 'n': v['n'],
            })
    df = pd.DataFrame(records).sort_values('pos_from_cds_start')
    print(f"Saliency aggregated: {len(df)} positions from {valid_count} samples")
    return df


# ============================================================
# Phase 1C: AdaLN gene attribution
# ============================================================
def compute_adaLN_gene_attribution(model, gene_names=None, top_k=50):
    """Pure weight inspection. Zero forward passes."""
    raw = _unwrap(model)
    n_layers = len(raw.encoder.encoder_layers)
    d_expr = raw.d_expr

    W_proj1 = raw.expr_projector[1].weight.detach().cpu().numpy()
    W_proj1_expr = np.abs(W_proj1[:, :d_expr])
    W_proj2 = np.abs(raw.expr_projector[3].weight.detach().cpu().numpy())

    all_attr = []
    for layer_idx in range(n_layers):
        for sub_idx, sub_name in [(0, 'attn'), (1, 'ffn')]:
            mod = raw.encoder.encoder_layers[layer_idx].sublayers[sub_idx].adaLN_modulation[1]
            W_ada = np.abs(mod.weight.detach().cpu().numpy())
            ada_imp = W_ada.sum(axis=0)
            gene_scores = ada_imp @ W_proj2 @ W_proj1_expr

            top_idx = np.argsort(gene_scores)[::-1][:top_k]
            for gi in top_idx:
                name = gene_names[gi] if gene_names and gi < len(gene_names) else f"GENE_{gi}"
                all_attr.append({
                    'layer': layer_idx,
                    'layer_module': f'L{layer_idx}-{sub_name}',
                    'gene': name, 'gene_idx': gi,
                    'score': float(gene_scores[gi]),
                })

    df = pd.DataFrame(all_attr)
    df['score_norm'] = df.groupby('layer_module')['score'].transform(lambda x: x / x.max())
    print(f"Gene attribution: {len(df)} entries, {df['gene'].nunique()} genes")
    return df


# ============================================================
# Hotspot identification from Phase 1
# ============================================================
def identify_hotspot_positions(attn_df, saliency_df, n_hotspots=30,
                                layer_range=None):
    """Combine attention + saliency Z-scores to select hotspots."""
    if layer_range is None:
        attn_agg = attn_df.groupby('pos_from_cds_start')['mean_attn'].mean().reset_index()
    else:
        mask = attn_df['layer'].between(*layer_range)
        attn_agg = attn_df[mask].groupby('pos_from_cds_start')['mean_attn'].mean().reset_index()

    merged = attn_agg.merge(saliency_df, on='pos_from_cds_start', how='outer').fillna(0)
    merged['attn_z'] = (merged['mean_attn'] - merged['mean_attn'].mean()) / (merged['mean_attn'].std() + 1e-8)
    merged['sal_z'] = (merged['mean_saliency'] - merged['mean_saliency'].mean()) / (merged['mean_saliency'].std() + 1e-8)
    merged['combined_score'] = (merged['attn_z'] + merged['sal_z']) / 2

    hotspots = merged.nlargest(n_hotspots, 'combined_score')
    positions = sorted(hotspots['pos_from_cds_start'].astype(int).tolist())
    print(f"Hotspot positions: {positions}")
    return positions, hotspots


# ============================================================
# Phase 2: Targeted mutagenesis at hotspots
# ============================================================
def targeted_mutagenesis(model, dataset, seq_dict, tx_cds,
                          target_positions, n_transcripts=30,
                          cell_type=None, device=None):
    """
    Only mutate K hotspot positions. Much faster than full sliding window.
    """
    raw = _unwrap(model)
    if device is None:
        device = next(raw.parameters()).device
    raw.eval()

    nt_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    valid_tids = [t for t in seq_dict if t in tx_cds
                  and tx_cds[t].get('cds_start_pos', -1) > 0]
    if len(valid_tids) > n_transcripts:
        selected = np.random.choice(valid_tids, n_transcripts, replace=False)
    else:
        selected = valid_tids

    results = []
    for tid in tqdm(selected, desc="Targeted mutagenesis"):
        cds_info = tx_cds[tid]
        cds_start = cds_info.get('cds_start_pos', -1) - 1
        cds_end = cds_info.get('cds_end_pos', -1) - 1
        if cds_start < 0 or cds_end <= cds_start:
            continue

        seq = seq_dict[tid].upper()
        L = len(seq)

        # Find matching dataset sample
        sample_idx = None
        for i in range(len(dataset)):
            tid_i = str(dataset[i][0]).rsplit('-', 2)[0]
            if tid_i == tid:
                if cell_type is None or dataset[i][2] == cell_type:
                    sample_idx = i
                    break
        if sample_idx is None:
            continue

        s = _extract_sample(dataset, sample_idx)
        se_ref = torch.from_numpy(s['se']).float().unsqueeze(0).to(device)
        ce_ref = torch.from_numpy(s['ce']).float().unsqueeze(0).to(device)
        if s['ev'] is not None and len(s['ev']) > 0:
            ev_ref = torch.from_numpy(s['ev']).float().unsqueeze(0).to(device)
        else:
            ev_ref = None

        # Baseline TE via model.predict
        with torch.no_grad():
            out_ref = model.predict(
                seq_batch=se_ref, count_batch=ce_ref, expr_vector=ev_ref,
                species=s['species'], head_names=['count'], return_numpy=False,
            )
            pred_ref = out_ref['count']
            if isinstance(pred_ref, dict):
                pred_ref = pred_ref.get('profile', pred_ref)
            te_ref = pred_ref[0, cds_start:cds_end:3, 0].mean().item()

        for rel_pos in target_positions:
            abs_pos = cds_start + rel_pos
            if abs_pos < 0 or abs_pos >= L:
                continue
            orig_base = seq[abs_pos]
            if orig_base not in nt_map:
                continue
            orig_idx = nt_map[orig_base]

            for tgt_base, tgt_idx in nt_map.items():
                if tgt_idx == orig_idx:
                    continue

                se_mut = se_ref.clone()
                se_mut[0, abs_pos, :] = 0
                se_mut[0, abs_pos, tgt_idx] = 1.0

                with torch.no_grad():
                    out_mut = model.predict(
                        seq_batch=se_mut, count_batch=ce_ref, expr_vector=ev_ref,
                        species=s['species'], head_names=['count'], return_numpy=False,
                    )
                    pred_mut = out_mut['count']
                    if isinstance(pred_mut, dict):
                        pred_mut = pred_mut.get('profile', pred_mut)
                    te_mut = pred_mut[0, cds_start:cds_end:3, 0].mean().item()

                results.append({
                    'tid': tid,
                    'pos_from_cds_start': rel_pos,
                    'ref_base': orig_base, 'mut_base': tgt_base,
                    'te_ref': te_ref, 'te_mut': te_mut,
                    'delta_te': te_mut - te_ref,
                    'log2_fc': np.log2((te_mut + 1e-8) / (te_ref + 1e-8)),
                })

    df = pd.DataFrame(results)
    n_mut = len(df)
    print(f"Targeted mutagenesis: {n_mut} mutations "
          f"({len(selected)} transcripts x {len(target_positions)} pos x 3 bases)")
    return df


# ============================================================
# Aggregate mutagenesis results
# ============================================================
def aggregate_mutagenesis(mut_df):
    pos_agg = mut_df.groupby('pos_from_cds_start').agg(
        mean_abs_delta=('delta_te', lambda x: np.abs(x).mean()),
        std_abs_delta=('delta_te', lambda x: np.abs(x).std()),
        max_abs_delta=('delta_te', lambda x: np.abs(x).max()),
        n=('delta_te', 'count'),
    ).reset_index()
    pos_agg['sem'] = pos_agg['std_abs_delta'] / np.sqrt(pos_agg['n'])

    base_agg = mut_df.groupby(['pos_from_cds_start', 'ref_base', 'mut_base']).agg(
        mean_delta=('delta_te', 'mean'),
        n=('delta_te', 'count'),
    ).reset_index()
    return pos_agg, base_agg


# ============================================================
# Sequence context extraction for MEME
# ============================================================
def extract_hotspot_contexts(seq_dict, tx_cds, hotspot_positions,
                              context_radius=20, max_seqs=200):
    contexts = defaultdict(list)
    tids = [t for t in seq_dict if t in tx_cds
            and tx_cds[t].get('cds_start_pos', -1) > 0]

    for tid in tids[:max_seqs]:
        cds_start = tx_cds[tid].get('cds_start_pos', -1) - 1
        seq = seq_dict[tid].upper()
        for rel_pos in hotspot_positions:
            abs_pos = cds_start + rel_pos
            if 0 <= abs_pos < len(seq):
                ctx_s = max(0, abs_pos - context_radius)
                ctx_e = min(len(seq), abs_pos + context_radius + 1)
                ctx = seq[ctx_s:ctx_e]
                if 'N' not in ctx:
                    contexts[rel_pos].append(ctx)
    return dict(contexts)


# ============================================================
# Plotting utilities
# ============================================================
def plot_attention_profile(attn_df, out_path, cds_region=(-50, 600),
                           stop_region=(-600, 50), layer_range=None):
    """
    Two-panel scatter plot of attention positional importance:
      Left:  aligned to CDS start (pos 0)
      Right: aligned to CDS stop  (pos 0 = last nt of CDS / first nt of 3'UTR)

    CDS region is shaded light grey. Points have alpha=0.4.

    Args:
        attn_df: from extract_attention_positional_importance
        out_path: output PDF path
        cds_region: (start_offset, end_offset) relative to CDS start
        stop_region: (start_offset, end_offset) relative to CDS stop
        layer_range: (min_layer, max_layer) to filter; None = all layers
    """
    from plotnine import ggplot, aes, geom_point, geom_rect, annotate, labs, theme_bw, theme, element_text, facet_wrap

    df = attn_df.copy()
    if layer_range is not None:
        df = df[df['layer'].between(*layer_range)]

    # --- Left panel: aligned to CDS start ---
    df_start = df[(df['pos_from_cds_start'] >= cds_region[0]) &
                  (df['pos_from_cds_start'] <= cds_region[1])].copy()
    df_start['panel'] = 'CDS start region'
    df_start['x_pos'] = df_start['pos_from_cds_start']

    # CDS shading: CDS starts at pos 0, ends at some average distance
    # We need the average CDS end from the data—use typical CDS length ~1000
    cds_shade_start = pd.DataFrame({
        'xmin': [0], 'xmax': [1000], 'panel': ['CDS start region']
    })

    # --- Right panel: aligned to CDS stop ---
    # pos_from_cds_start relative to start. To get relative to stop:
    # We need cds_end info. Since attn_df only has pos_from_cds_start,
    # we create a mirror: the stop region view shows positions that are
    # near the typical CDS end. We approximate: positions where
    # pos_from_cds_start is in the CDS range map to stop-aligned.
    # Better: use a separate input or compute from dataset.
    # For now, we use the same data but shifted: stop_pos = pos - typical_cds_len
    typical_cds_len = 800  # approximate; adjust based on your data
    df_stop = df[(df['pos_from_cds_start'] >= stop_region[0] + typical_cds_len) &
                 (df['pos_from_cds_start'] <= stop_region[1] + typical_cds_len)].copy()
    if len(df_stop) == 0:
        # Fallback: just use the tail of the start-aligned data
        df_stop = df[(df['pos_from_cds_start'] >= -stop_region[1]) &
                     (df['pos_from_cds_start'] <= -stop_region[0])].copy()
        df_stop['x_pos'] = -df_stop['pos_from_cds_start']  # flip
    else:
        df_stop['x_pos'] = df_stop['pos_from_cds_start'] - typical_cds_len
    df_stop['panel'] = 'CDS stop region'

    cds_shade_stop = pd.DataFrame({
        'xmin': [-typical_cds_len], 'xmax': [0], 'panel': ['CDS stop region']
    })

    df_plot = pd.concat([df_start, df_stop], ignore_index=True)
    df_plot['panel'] = pd.Categorical(df_plot['panel'],
                                       categories=['CDS start region', 'CDS stop region'])

    cds_shade = pd.concat([cds_shade_start, cds_shade_stop], ignore_index=True)
    cds_shade['panel'] = pd.Categorical(cds_shade['panel'],
                                         categories=['CDS start region', 'CDS stop region'])

    p = (
        ggplot(df_plot, aes(x='x_pos', y='mean_attn'))
        + geom_rect(data=cds_shade,
                    mapping=aes(xmin='xmin', xmax='xmax'),
                    ymin=-float('inf'), ymax=float('inf'),
                    fill='#E0E0E0', alpha=0.5, inherit_aes=False)
        + geom_point(alpha=0.4, size=0.8, color='#2171B5')
        + facet_wrap('~panel', scales='free_x', nrow=1)
        + geom_vline(xintercept=0, linetype='dashed', color='#C44E52', size=0.6)
        + labs(x='Position relative to CDS boundary (nt)', y='Mean attention received')
        + theme_bw()
        + theme(figure_size=(16, 5))
    )
    p.save(out_path)
    print(f"Attention profile saved to {out_path}")


def plot_saliency_profile(sal_df, out_path, cds_region=(-50, 600),
                           stop_region=(-600, 50)):
    """
    Two-panel scatter plot of input saliency, same layout as attention.
    """
    from plotnine import ggplot, aes, geom_point, geom_rect, geom_vline, labs, theme_bw, theme, facet_wrap
    import pandas as pd

    df = sal_df.copy()
    typical_cds_len = 800

    # Left: CDS start
    df_start = df[(df['pos_from_cds_start'] >= cds_region[0]) &
                  (df['pos_from_cds_start'] <= cds_region[1])].copy()
    df_start['panel'] = 'CDS start region'
    df_start['x_pos'] = df_start['pos_from_cds_start']

    # Right: CDS stop
    df_stop = df[(df['pos_from_cds_start'] >= stop_region[0] + typical_cds_len) &
                 (df['pos_from_cds_start'] <= stop_region[1] + typical_cds_len)].copy()
    if len(df_stop) == 0:
        df_stop = df[(df['pos_from_cds_start'] >= -stop_region[1]) &
                     (df['pos_from_cds_start'] <= -stop_region[0])].copy()
        df_stop['x_pos'] = -df_stop['pos_from_cds_start']
    else:
        df_stop['x_pos'] = df_stop['pos_from_cds_start'] - typical_cds_len
    df_stop['panel'] = 'CDS stop region'

    df_plot = pd.concat([df_start, df_stop], ignore_index=True)
    df_plot['panel'] = pd.Categorical(df_plot['panel'],
                                       categories=['CDS start region', 'CDS stop region'])

    cds_shade = pd.DataFrame({
        'xmin': [0, -typical_cds_len],
        'xmax': [typical_cds_len, 0],
        'panel': ['CDS start region', 'CDS stop region']
    })
    cds_shade['panel'] = pd.Categorical(cds_shade['panel'],
                                         categories=['CDS start region', 'CDS stop region'])

    p = (
        ggplot(df_plot, aes(x='x_pos', y='mean_saliency'))
        + geom_rect(data=cds_shade,
                    mapping=aes(xmin='xmin', xmax='xmax'),
                    ymin=-float('inf'), ymax=float('inf'),
                    fill='#E0E0E0', alpha=0.5, inherit_aes=False)
        + geom_point(alpha=0.4, size=0.8, color='#238B45')
        + facet_wrap('~panel', scales='free_x', nrow=1)
        + geom_vline(xintercept=0, linetype='dashed', color='#C44E52', size=0.6)
        + labs(x='Position relative to CDS boundary (nt)', y='Mean |d(TE)/d(base)|')
        + theme_bw()
        + theme(figure_size=(16, 5))
    )
    p.save(out_path)
    print(f"Saliency profile saved to {out_path}")


def plot_mutagenesis_profile(pos_agg, out_path, cds_region=(-50, 600),
                              stop_region=(-600, 50)):
    """
    Two-panel scatter plot of mutagenesis impact (mean |delta_TE|).
    """
    from plotnine import ggplot, aes, geom_point, geom_rect, geom_vline, labs, theme_bw, theme, facet_wrap

    df = pos_agg.copy()
    typical_cds_len = 800

    # Left: CDS start
    df_start = df[(df['pos_from_cds_start'] >= cds_region[0]) &
                  (df['pos_from_cds_start'] <= cds_region[1])].copy()
    df_start['panel'] = 'CDS start region'
    df_start['x_pos'] = df_start['pos_from_cds_start']

    # Right: CDS stop
    df_stop = df[(df['pos_from_cds_start'] >= stop_region[0] + typical_cds_len) &
                 (df['pos_from_cds_start'] <= stop_region[1] + typical_cds_len)].copy()
    if len(df_stop) == 0:
        df_stop = df[(df['pos_from_cds_start'] >= -stop_region[1]) &
                     (df['pos_from_cds_start'] <= -stop_region[0])].copy()
        df_stop['x_pos'] = -df_stop['pos_from_cds_start']
    else:
        df_stop['x_pos'] = df_stop['pos_from_cds_start'] - typical_cds_len
    df_stop['panel'] = 'CDS stop region'

    df_plot = pd.concat([df_start, df_stop], ignore_index=True)
    df_plot['panel'] = pd.Categorical(df_plot['panel'],
                                       categories=['CDS start region', 'CDS stop region'])

    cds_shade = pd.DataFrame({
        'xmin': [0, -typical_cds_len],
        'xmax': [typical_cds_len, 0],
        'panel': ['CDS start region', 'CDS stop region']
    })
    cds_shade['panel'] = pd.Categorical(cds_shade['panel'],
                                         categories=['CDS start region', 'CDS stop region'])

    p = (
        ggplot(df_plot, aes(x='x_pos', y='mean_abs_delta'))
        + geom_rect(data=cds_shade,
                    mapping=aes(xmin='xmin', xmax='xmax'),
                    ymin=-float('inf'), ymax=float('inf'),
                    fill='#E0E0E0', alpha=0.5, inherit_aes=False)
        + geom_point(alpha=0.4, size=0.8, color='#8C2D04')
        + facet_wrap('~panel', scales='free_x', nrow=1)
        + geom_vline(xintercept=0, linetype='dashed', color='#C44E52', size=0.6)
        + labs(x='Position relative to CDS boundary (nt)', y='Mean |Delta TE|')
        + theme_bw()
        + theme(figure_size=(16, 5))
    )
    p.save(out_path)
    print(f"Mutagenesis profile saved to {out_path}")
