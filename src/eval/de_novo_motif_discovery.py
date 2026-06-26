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
    aggregate aligned to BOTH CDS start and CDS stop.

    Returns:
        attn_df: DataFrame with columns:
            layer, pos_from_cds_start, pos_from_cds_stop,
            mean_attn, std_attn, n_contrib
        avg_cds_len: float, median CDS length across sampled transcripts
    """
    raw = _unwrap(model)
    if device is None:
        device = next(raw.parameters()).device
    raw.eval()

    n_layers = len(raw.encoder.encoder_layers)
    n_heads = raw.n_heads
    head_dim = raw.encoder.encoder_layers[0].multi_headed_attention.head_dim

    accum_start = defaultdict(lambda: {'sum': 0.0, 'sum_sq': 0.0, 'n': 0})
    accum_stop  = defaultdict(lambda: {'sum': 0.0, 'sum_sq': 0.0, 'n': 0})
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    valid_count = 0
    cds_lengths = []  # collect real CDS lengths

    for idx in tqdm(indices, desc="Attention positional importance"):
        s = _extract_sample(dataset, idx)
        if not s['valid'] or s['L'] > max_len:
            continue

        se = torch.from_numpy(s['se']).float().unsqueeze(0).to(device)
        ce = torch.from_numpy(s['ce']).float().unsqueeze(0).to(device)
        if s['ev'] is not None and len(s['ev']) > 0:
            ev = torch.from_numpy(s['ev']).float().unsqueeze(0).to(device)
        else:
            ev = None
        L = s['L']
        cds_start = s['cds_start_0']
        cds_end = s['cds_end_0']
        cds_len = cds_end - cds_start
        cds_lengths.append(cds_len)

        with torch.no_grad():
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

                received = attn_w.sum(dim=2).mean(dim=1)[0].cpu().numpy()

                for pos in range(Lc):
                    rel_start = pos - cds_start
                    rel_stop = pos - cds_end

                    accum_start[(layer_idx, rel_start)]['sum'] += float(received[pos])
                    accum_start[(layer_idx, rel_start)]['sum_sq'] += float(received[pos]) ** 2
                    accum_start[(layer_idx, rel_start)]['n'] += 1

                    accum_stop[(layer_idx, rel_stop)]['sum'] += float(received[pos])
                    accum_stop[(layer_idx, rel_stop)]['sum_sq'] += float(received[pos]) ** 2
                    accum_stop[(layer_idx, rel_stop)]['n'] += 1

                # Complete sublayer
                attn_out = torch.matmul(attn_w, v)
                attn_out = attn_out.transpose(1, 2).reshape(bs_, Lc, n_heads * head_dim)
                attn_out = attn_mod.unifyheads(attn_out)
                if hasattr(attn_mod, 'dropout'):
                    attn_out = attn_mod.dropout(attn_out)
                src_reps = src_reps + alpha.unsqueeze(1) * sub.dropout(attn_out)

                # FFN
                sub2 = enc_layer.sublayers[1]
                style2 = sub2.adaLN_modulation(compact_style)
                gamma2, beta2, alpha2 = style2.chunk(3, dim=-1)
                normed2 = (1 + gamma2.unsqueeze(1)) * sub2.LN(src_reps) + beta2.unsqueeze(1)
                ffn_out = sub2.dropout(enc_layer.ffn(normed2))
                src_reps = src_reps + alpha2.unsqueeze(1) * ffn_out

        valid_count += 1

    avg_cds_len = float(np.median(cds_lengths)) if cds_lengths else 800.0

    records = []
    for (layer, pos), v in accum_start.items():
        if v['n'] >= 5:
            mean = v['sum'] / v['n']
            std = np.sqrt(max(0, v['sum_sq'] / v['n'] - mean ** 2))
            records.append({
                'layer': layer,
                'pos_from_cds_start': pos,
                'mean_attn': mean,
                'std_attn': std,
                'n_contrib': v['n'],
            })

    df_start = pd.DataFrame(records)

    records_stop = []
    for (layer, pos), v in accum_stop.items():
        if v['n'] >= 5:
            mean = v['sum'] / v['n']
            std = np.sqrt(max(0, v['sum_sq'] / v['n'] - mean ** 2))
            records_stop.append({
                'layer': layer,
                'pos_from_cds_stop': pos,
                'mean_attn': mean,
                'std_attn': std,
                'n_contrib': v['n'],
            })

    df_stop = pd.DataFrame(records_stop)

    # Merge into a single DataFrame with both alignments
    # Each row has: layer, pos_from_cds_start, pos_from_cds_stop, mean_attn_start, mean_attn_stop, ...
    # We use two separate DataFrames for simplicity; plotting functions handle them.
    print(f"Attention aggregated: {len(df_start)} start-aligned + {len(df_stop)} stop-aligned entries "
          f"from {valid_count} samples (median CDS={avg_cds_len:.0f} nt)")
    return df_start, df_stop, avg_cds_len
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

    accum_start = defaultdict(lambda: {'sum': 0.0, 'sum_sq': 0.0, 'n': 0})
    accum_stop = defaultdict(lambda: {'sum': 0.0, 'sum_sq': 0.0, 'n': 0})
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
            rel_start = pos - cds_start
            rel_stop = pos - cds_end
            accum_start[rel_start]['sum'] += float(sal[pos])
            accum_start[rel_start]['sum_sq'] += float(sal[pos]) ** 2
            accum_start[rel_start]['n'] += 1
            accum_stop[rel_stop]['sum'] += float(sal[pos])
            accum_stop[rel_stop]['sum_sq'] += float(sal[pos]) ** 2
            accum_stop[rel_stop]['n'] += 1

        se.grad = None
        valid_count += 1

    records_start = []
    for pos, v in accum_start.items():
        if v['n'] >= 5:
            mean = v['sum'] / v['n']
            std = np.sqrt(max(0, v['sum_sq'] / v['n'] - mean ** 2))
            records_start.append({
                'pos_from_cds_start': pos,
                'pos_from_cds_stop': float('nan'),
                'mean_saliency': mean, 'std_saliency': std, 'n': v['n'],
            })

    records_stop = []
    for pos, v in accum_stop.items():
        if v['n'] >= 5:
            mean = v['sum'] / v['n']
            std = np.sqrt(max(0, v['sum_sq'] / v['n'] - mean ** 2))
            records_stop.append({
                'pos_from_cds_start': float('nan'),
                'pos_from_cds_stop': pos,
                'mean_saliency': mean, 'std_saliency': std, 'n': v['n'],
            })

    # Merge: each row gets both alignments when available; start-aligned row
    # carries the mean_saliency, and stop-aligned entries are separate rows
    # so plot_saliency_profile can read pos_from_cds_stop column directly.
    # We return a single df with both pos columns; rows with valid pos_from_cds_start
    # are for start panel, rows with valid pos_from_cds_stop are for stop panel.
    df_start = pd.DataFrame(records_start).sort_values('pos_from_cds_start')
    # For stop rows, we also want a pos_from_cds_start column filled with nan
    # so concat works; already handled above.
    df_all = df_start.copy()

    if records_stop:
        df_stop = pd.DataFrame(records_stop).sort_values('pos_from_cds_stop')
        # For the stop panel, we map pos_from_cds_stop back to pos_from_cds_start
        # using avg_cds_len (approximation), so identify_hotspot_positions still works
        # with pos_from_cds_start. But the plotting function prefers pos_from_cds_stop.
        df_all = pd.concat([df_start, df_stop], ignore_index=True)

    print(f"Saliency aggregated: {len(df_start)} start-aligned + "
          f"{len(df_stop) if records_stop else 0} stop-aligned entries "
          f"from {valid_count} samples")
    return df_all


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
# Plotting utilities — CDS-start and CDS-stop aligned, per-layer color
# ============================================================
def plot_attention_profile(df_start, df_stop, out_path="attention_profile.pdf",
                            start_region=(-100, 300), stop_region=(-300, 100)):
    """
    Two-panel per-layer attention profile.

    Left:  aligned to CDS start (pos 0 = first nt of start codon)
    Right: aligned to CDS stop  (pos 0 = first nt of stop codon, first nt of 3'UTR)

    Each layer is a different color with LOESS smooth curve.
    CDS region is shaded light grey.

    Args:
        df_start: DataFrame from extract_attention_positional_importance,
                  column 'pos_from_cds_start'
        df_stop:  DataFrame from extract_attention_positional_importance,
                  column 'pos_from_cds_stop'
        out_path: output PDF path
        start_region, stop_region: (left, right) nt relative to CDS boundary
    """
    from plotnine import (ggplot, aes, geom_point, geom_smooth, geom_rect,
                          geom_vline, labs, theme_bw, theme,
                          scale_color_manual, facet_wrap)
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    n_layers = df_start['layer'].nunique()
    cmap = cm.get_cmap('viridis', n_layers)
    layer_colors = {i: mcolors.to_hex(cmap(i)) for i in range(n_layers)}

    # --- Left panel: CDS start ---
    df_s = df_start[
        df_start['pos_from_cds_start'].between(*start_region)
    ].copy()
    df_s['panel'] = 'CDS start'
    df_s['x_pos'] = df_s['pos_from_cds_start']

    # --- Right panel: CDS stop (real coordinates from cds_end) ---
    df_e = df_stop[
        df_stop['pos_from_cds_stop'].between(*stop_region)
    ].copy()
    df_e['panel'] = 'CDS stop'
    df_e['x_pos'] = df_e['pos_from_cds_stop']

    df_plot = pd.concat([df_s, df_e], ignore_index=True)
    df_plot['panel'] = pd.Categorical(df_plot['panel'], categories=['CDS start', 'CDS stop'])
    df_plot['layer'] = df_plot['layer'].astype(int)

    # CDS shading: for start panel CDS is to the right of zero,
    # for stop panel CDS is to the left of zero
    cds_shade = pd.DataFrame({
        'xmin': [0, stop_region[0]],
        'xmax': [start_region[1], 0],
        'panel': ['CDS start', 'CDS stop'],
    })
    cds_shade['panel'] = pd.Categorical(cds_shade['panel'], categories=['CDS start', 'CDS stop'])

    p = (
        ggplot(df_plot, aes(x='x_pos', y='mean_attn', color='factor(layer)', group='layer'))
        + geom_rect(data=cds_shade,
                    mapping=aes(xmin='xmin', xmax='xmax'),
                    ymin=-float('inf'), ymax=float('inf'),
                    fill='#E8E8E8', alpha=0.4, color=None, inherit_aes=False)
        + geom_point(alpha=0.3, size=0.6, stroke=0)
        + geom_smooth(method='loess', span=0.3, se=False, size=0.8)
        + facet_wrap('~panel', scales='free_x', nrow=1)
        + geom_vline(xintercept=0, linetype='dashed', color='#C44E52', size=0.6)
        + scale_color_manual(values=layer_colors, name='Layer')
        + labs(x='Position relative to CDS boundary (nt)', y='Mean attention received')
        + theme_bw()
        + theme(figure_size=(18, 6), legend_position='right')
    )
    p.save(out_path)
    print(f"Attention profile saved to {out_path}")


def plot_saliency_profile(sal_df, avg_cds_len=800, out_path="saliency_profile.pdf",
                           start_region=(-100, 300), stop_region=(-300, 100)):
    """
    Two-panel saliency scatter with CDS shading.
    Uses real CDS coordinates from sal_df: cds_start and cds_end per sample.

    Args:
        sal_df: from compute_saliency_profile, with columns:
                pos_from_cds_start, pos_from_cds_stop, mean_saliency, ...
        avg_cds_len: fallback CDS length if cds_end not available per row
        out_path: output PDF
        start_region, stop_region: (left, right) relative to CDS boundary
    """
    from plotnine import (ggplot, aes, geom_point, geom_rect, geom_vline,
                          labs, theme_bw, theme, facet_wrap)

    df = sal_df.copy()

    df_start = df[df['pos_from_cds_start'].between(*start_region)].copy()
    df_start['panel'] = 'CDS start'
    df_start['x_pos'] = df_start['pos_from_cds_start']

    if 'pos_from_cds_stop' in df.columns:
        df_stop = df[df['pos_from_cds_stop'].between(*stop_region)].copy()
        df_stop['panel'] = 'CDS stop'
        df_stop['x_pos'] = df_stop['pos_from_cds_stop']
    else:
        df_stop = df[df['pos_from_cds_start'].between(
            stop_region[0] + avg_cds_len, stop_region[1] + avg_cds_len
        )].copy()
        df_stop['panel'] = 'CDS stop'
        df_stop['x_pos'] = df_stop['pos_from_cds_start'] - avg_cds_len

    df_plot = pd.concat([df_start, df_stop], ignore_index=True)
    df_plot['panel'] = pd.Categorical(df_plot['panel'], categories=['CDS start', 'CDS stop'])

    cds_shade = pd.DataFrame({
        'xmin': [0, stop_region[0]],
        'xmax': [start_region[1], 0],
        'panel': ['CDS start', 'CDS stop'],
    })
    cds_shade['panel'] = pd.Categorical(cds_shade['panel'], categories=['CDS start', 'CDS stop'])

    p = (
        ggplot(df_plot, aes(x='x_pos', y='mean_saliency'))
        + geom_rect(data=cds_shade,
                    mapping=aes(xmin='xmin', xmax='xmax'),
                    ymin=-float('inf'), ymax=float('inf'),
                    fill='#E8E8E8', alpha=0.4, color=None, inherit_aes=False)
        + geom_point(alpha=0.4, size=0.8, color='#238B45', stroke=0)
        + facet_wrap('~panel', scales='free_x', nrow=1)
        + geom_vline(xintercept=0, linetype='dashed', color='#C44E52', size=0.6)
        + labs(x='Position relative to CDS boundary (nt)', y='Mean |d(TE)/d(base)|')
        + theme_bw()
        + theme(figure_size=(18, 5))
    )
    p.save(out_path)
    print(f"Saliency profile saved to {out_path}")


def plot_mutagenesis_profile(pos_agg, avg_cds_len=800, out_path="mutagenesis_profile.pdf",
                              start_region=(-100, 300), stop_region=(-300, 100)):
    """
    Two-panel mutagenesis impact scatter.

    Args:
        pos_agg: from aggregate_mutagenesis, column 'pos_from_cds_start'
        avg_cds_len: estimated CDS length for stop alignment
        out_path: output PDF
        start_region, stop_region: (left, right) relative to CDS boundary
    """
    from plotnine import (ggplot, aes, geom_point, geom_rect, geom_vline,
                          labs, theme_bw, theme, facet_wrap)

    df = pos_agg.copy()

    df_start = df[df['pos_from_cds_start'].between(*start_region)].copy()
    df_start['panel'] = 'CDS start'
    df_start['x_pos'] = df_start['pos_from_cds_start']

    df_stop = df[df['pos_from_cds_start'].between(
        stop_region[0] + avg_cds_len, stop_region[1] + avg_cds_len
    )].copy()
    df_stop['panel'] = 'CDS stop'
    df_stop['x_pos'] = df_stop['pos_from_cds_start'] - avg_cds_len

    df_plot = pd.concat([df_start, df_stop], ignore_index=True)
    df_plot['panel'] = pd.Categorical(df_plot['panel'], categories=['CDS start', 'CDS stop'])

    cds_shade = pd.DataFrame({
        'xmin': [0, stop_region[0]],
        'xmax': [start_region[1], 0],
        'panel': ['CDS start', 'CDS stop'],
    })
    cds_shade['panel'] = pd.Categorical(cds_shade['panel'], categories=['CDS start', 'CDS stop'])

    p = (
        ggplot(df_plot, aes(x='x_pos', y='mean_abs_delta'))
        + geom_rect(data=cds_shade,
                    mapping=aes(xmin='xmin', xmax='xmax'),
                    ymin=-float('inf'), ymax=float('inf'),
                    fill='#E8E8E8', alpha=0.4, color=None, inherit_aes=False)
        + geom_point(alpha=0.4, size=0.8, color='#8C2D04', stroke=0)
        + facet_wrap('~panel', scales='free_x', nrow=1)
        + geom_vline(xintercept=0, linetype='dashed', color='#C44E52', size=0.6)
        + labs(x='Position relative to CDS boundary (nt)', y='Mean |Delta TE|')
        + theme_bw()
        + theme(figure_size=(18, 5))
    )
    p.save(out_path)
    print(f"Mutagenesis profile saved to {out_path}")
