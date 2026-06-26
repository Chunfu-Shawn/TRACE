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
def _load_gene_names(gene_order_path=None, gene_annot_path=None):
    """
    Build gene name list aligned with expr_array column order.

    Args:
        gene_order_path: path to file listing ENSGs in expr_array column order
                         (default: src/config/global_anchor_gene_order.txt relative to TRACE)
        gene_annot_path: optional path to ENSG→gene_name mapping TSV
                         (columns: Gene stable ID, Transcript stable ID, Gene name, ...)

    Returns:
        list of gene name strings, same length as gene_order file.
    """
    import os as _os
    if gene_order_path is None:
        gene_order_path = _os.path.join(_os.path.dirname(__file__), '..', '..',
                                        'src', 'config', 'global_anchor_gene_order.txt')
    with open(gene_order_path) as f:
        ensg_list = [line.strip() for line in f if line.strip()]

    # Build ENSG → gene_name map if annotation file provided
    ensg2name = {}
    if gene_annot_path is not None:
        with open(gene_annot_path) as f:
            header = f.readline().strip().split('\t')
            # Expected columns: Gene stable ID, Transcript stable ID, Gene name, ...
            gid_col = header.index('Gene stable ID') if 'Gene stable ID' in header else 0
            gname_col = header.index('Gene name') if 'Gene name' in header else 2
            for line in f:
                cols = line.strip().split('\t')
                if len(cols) > max(gid_col, gname_col):
                    ensg2name[cols[gid_col]] = cols[gname_col]

    # Map ENSG → gene_name; fallback to ENSG ID itself
    gene_names = [ensg2name.get(e, e) for e in ensg_list]
    print(f"Loaded {len(gene_names)} gene names "
          f"({sum(1 for g in gene_names if g not in ensg_list)} with gene symbols)")
    return gene_names


def compute_adaLN_gene_attribution(model, gene_names=None, top_k=50,
                                    gene_annot_path=None):
    """
    Pure weight inspection — zero forward passes.

    Traces gene expression influence through the cell-environment projector:
      expr_array (16840) → Linear → LayerNorm → GELU → Linear → adaptive_dim
    then into each AdaLN modulation layer, to rank genes by total contribution.

    Args:
        model: TranslationBaseModel (or DDP-wrapped).
        gene_names: list of gene names, length = d_expr. If None, auto-loaded
                    from src/config/global_anchor_gene_order.txt.
        top_k: number of top genes per layer-module to record.
        gene_annot_path: path to ENSG→gene_name TSV (e.g. ens_genes_v112.txt).
                         If provided, gene_names will display gene symbols.

    Returns:
        DataFrame with columns: layer, layer_module, gene, gene_idx, score, score_norm
    """
    raw = _unwrap(model)
    n_layers = len(raw.encoder.encoder_layers)
    d_expr = raw.d_expr

    if gene_names is None:
        gene_names = _load_gene_names(gene_annot_path=gene_annot_path)

    W_proj1 = raw.expr_projector[1].weight.detach().cpu().numpy()
    W_proj1_expr = np.abs(W_proj1[:, :d_expr])
    W_proj2 = np.abs(raw.expr_projector[4].weight.detach().cpu().numpy())

    all_attr = []
    for layer_idx in range(n_layers):
        for sub_idx, sub_name in [(0, 'attn'), (1, 'ffn')]:
            mod = raw.encoder.encoder_layers[layer_idx].sublayers[sub_idx].adaLN_modulation[1]
            W_ada = np.abs(mod.weight.detach().cpu().numpy())
            ada_imp = W_ada.sum(axis=0)
            gene_scores = ada_imp @ W_proj2 @ W_proj1_expr

            top_idx = np.argsort(gene_scores)[::-1][:top_k]
            for gi in top_idx:
                name = gene_names[gi] if gi < len(gene_names) else f"GENE_{gi}"
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

def _cds_rect_data(start_region, stop_region):
    """Build geom_rect data for CDS shading and START/STOP codon annotation."""
    rect_cds = pd.DataFrame({
        'panel': ['CDS start', 'CDS start', 'CDS stop', 'CDS stop'],
        'xmin': [0, start_region[0], stop_region[0], -6],
        'xmax': [start_region[1], 0, 0, 0],
        'ymin': [-float('inf'), -float('inf'), -float('inf'), -float('inf')],
        'ymax': [float('inf'), float('inf'), float('inf'), float('inf')],
        'fill': ['#E8E8E8', '#F8F8F8', '#E8E8E8', '#D4B9B9'],
    })
    rect_cds['panel'] = pd.Categorical(rect_cds['panel'], categories=['CDS start', 'CDS stop'])
    return rect_cds


def plot_attention_profile(df_start, df_stop, out_path="attention_profile.pdf",
                            start_region=(-100, 300), stop_region=(-300, 100)):
    """
    Per-layer + combined attention profiles.

    - One figure per layer: facet_wrap CDS-start / CDS-stop.
    - One combined figure: all positions across layers, grey-blue scatter, no smoothing.
    CDS region shaded grey; START/STOP codon annotated.

    Args:
        df_start: from extract_attention_positional_importance, col 'pos_from_cds_start'
        df_stop:  from extract_attention_positional_importance, col 'pos_from_cds_stop'
        out_path: output PDF path (inserted before .pdf for layer / combined variants)
    """
    from plotnine import (ggplot, aes, geom_point, geom_rect, geom_vline,
                          labs, theme_bw, theme, facet_wrap, facet_grid, ylim)
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    n_layers = df_start['layer'].nunique()
    cmap = cm.get_cmap('viridis', n_layers)
    layer_colors = {i: mcolors.to_hex(cmap(i)) for i in range(n_layers)}

    # --- Prepare shared data ---
    df_s = df_start[df_start['pos_from_cds_start'].between(*start_region)].copy()
    df_s['panel'] = 'CDS start'
    df_s['x_pos'] = df_s['pos_from_cds_start']

    df_e = df_stop[df_stop['pos_from_cds_stop'].between(*stop_region)].copy()
    df_e['panel'] = 'CDS stop'
    df_e['x_pos'] = df_e['pos_from_cds_stop']

    df_plot = pd.concat([df_s, df_e], ignore_index=True)
    df_plot['panel'] = pd.Categorical(df_plot['panel'], categories=['CDS start', 'CDS stop'])
    df_plot['layer'] = df_plot['layer'].astype(int)

    rect_cds = _cds_rect_data(start_region, stop_region)

    base_out = out_path.replace('.pdf', '')

    # Clip y: use 99th percentile to avoid outlier-driven scale
    y_cap = df_plot['mean_attn'].quantile(0.99)

    # ============================================================
    # Combined: all layers pooled, grey-blue scatter, per-panel ylim
    # ============================================================
    # For combined, aggregate mean_attn across layers per position+panel
    df_combined = (df_plot
        .groupby(['x_pos', 'panel'], as_index=False)
        .agg(mean_attn=('mean_attn', 'mean')))
    y_cap_comb = df_combined['mean_attn'].quantile(0.99)

    p_comb = (
        ggplot(df_combined, aes(x='x_pos', y='mean_attn'))
        + geom_rect(data=rect_cds,
                    mapping=aes(xmin='xmin', xmax='xmax', ymin='ymin', ymax='ymax', fill='fill'),
                    alpha=0.3, inherit_aes=False, show_legend=False)
        + scale_fill_identity()
        + geom_point(alpha=0.35, size=0.7, color='#6A7B8B', stroke=0)
        + facet_wrap('~panel', scales='free_x', nrow=1)
        + geom_vline(xintercept=0, linetype='dashed', color='#C44E52', size=0.5)
        + ylim(0, y_cap_comb)
        + labs(x='Position relative to CDS boundary (nt)', y='Mean attention received')
        + theme_bw()
        + theme(figure_size=(16, 4))
    )
    p_comb.save(f"{base_out}.combined.pdf")
    print(f"Combined attention profile saved to {base_out}.combined.pdf")

    # ============================================================
    # Per-layer faceted: 12 layers in one figure, 2 columns (start/stop)
    # ============================================================
    df_plot_clipped = df_plot.copy()
    df_plot_clipped.loc[df_plot_clipped['mean_attn'] > y_cap, 'mean_attn'] = y_cap
    df_plot_clipped['Layer'] = pd.Categorical(
        [f'L{li}' for li in df_plot_clipped['layer']],
        categories=[f'L{i}' for i in range(n_layers)],
    )

    # Need separate rect_cds per layer-row for facet_grid to shade correctly
    n_panels = n_layers * 2  # layers x (start, stop)
    rect_per_layer = pd.DataFrame({
        'Layer': pd.Categorical(
            [f'L{i}' for i in range(n_layers) for _ in range(4)],
            categories=[f'L{i}' for i in range(n_layers)],
        ),
        'panel': pd.Categorical(
            rect_cds['panel'].tolist() * n_layers,
            categories=['CDS start', 'CDS stop'],
        ),
        'xmin': rect_cds['xmin'].tolist() * n_layers,
        'xmax': rect_cds['xmax'].tolist() * n_layers,
        'ymin': rect_cds['ymin'].tolist() * n_layers,
        'ymax': rect_cds['ymax'].tolist() * n_layers,
        'fill': rect_cds['fill'].tolist() * n_layers,
    })

    p_layers = (
        ggplot(df_plot_clipped, aes(x='x_pos', y='mean_attn'))
        + geom_rect(data=rect_per_layer,
                    mapping=aes(xmin='xmin', xmax='xmax', ymin='ymin', ymax='ymax', fill='fill'),
                    alpha=0.3, inherit_aes=False, show_legend=False)
        + scale_fill_identity()
        + geom_point(alpha=0.25, size=0.35, color='#6A7B8B', stroke=0)
        + facet_grid('Layer ~ panel', scales='free_x')
        + geom_vline(xintercept=0, linetype='dashed', color='#C44E52', size=0.4)
        + ylim(0, y_cap)
        + labs(x='Position relative to CDS boundary (nt)', y='Mean attention received')
        + theme_bw()
        + theme(figure_size=(16, 18), strip_text_y=element_text(size=7))
    )
    p_layers.save(f"{base_out}.per_layer.pdf")
    print(f"Per-layer attention faceted plot saved to {base_out}.per_layer.pdf")


def plot_saliency_profile(sal_df, avg_cds_len=800, out_path="saliency_profile.pdf",
                           start_region=(-100, 300), stop_region=(-300, 100)):
    """
    Two-panel saliency scatter with CDS shading.
    Y-axis in 10⁻³ scale.

    Args:
        sal_df: from compute_saliency_profile, columns pos_from_cds_start, pos_from_cds_stop, mean_saliency, ...
        avg_cds_len: fallback CDS length
        out_path: output PDF
        start_region, stop_region: (left, right) relative to CDS boundary
    """
    from plotnine import (ggplot, aes, geom_point, geom_rect, geom_vline,
                          labs, theme_bw, theme, facet_wrap, ylim)

    df = sal_df.copy()

    df_start = df[df['pos_from_cds_start'].between(*start_region)].copy()
    df_start['panel'] = 'CDS start'
    df_start['x_pos'] = df_start['pos_from_cds_start']

    if 'pos_from_cds_stop' in df.columns and df['pos_from_cds_stop'].notna().any():
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

    # Convert to 10^-3 scale
    df_plot['saliency_1e3'] = df_plot['mean_saliency'] * 1000

    rect_cds = _cds_rect_data(start_region, stop_region)
    y_cap = df_plot['saliency_1e3'].quantile(0.99)

    p = (
        ggplot(df_plot, aes(x='x_pos', y='saliency_1e3'))
        + geom_rect(data=rect_cds,
                    mapping=aes(xmin='xmin', xmax='xmax', ymin='ymin', ymax='ymax', fill='fill'),
                    alpha=0.3, inherit_aes=False, show_legend=False)
        + scale_fill_identity()
        + geom_point(alpha=0.4, size=0.8, color='#238B45', stroke=0)
        + facet_wrap('~panel', scales='free_x', nrow=1)
        + geom_vline(xintercept=0, linetype='dashed', color='#C44E52', size=0.5)
        + ylim(0, y_cap)
        + labs(x='Position relative to CDS boundary (nt)',
               y='Mean |d(TE)/d(base)| (×10⁻³)')
        + theme_bw()
        + theme(figure_size=(16, 4))
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
                          labs, theme_bw, theme, facet_wrap, ylim)

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

    rect_cds = _cds_rect_data(start_region, stop_region)
    y_cap = df_plot['mean_abs_delta'].quantile(0.99)

    p = (
        ggplot(df_plot, aes(x='x_pos', y='mean_abs_delta'))
        + geom_rect(data=rect_cds,
                    mapping=aes(xmin='xmin', xmax='xmax', ymin='ymin', ymax='ymax', fill='fill'),
                    alpha=0.3, inherit_aes=False, show_legend=False)
        + scale_fill_identity()
        + geom_point(alpha=0.4, size=0.8, color='#8C2D04', stroke=0)
        + facet_wrap('~panel', scales='free_x', nrow=1)
        + geom_vline(xintercept=0, linetype='dashed', color='#C44E52', size=0.5)
        + ylim(0, y_cap)
        + labs(x='Position relative to CDS boundary (nt)', y='Mean |Delta TE|')
        + theme_bw()
        + theme(figure_size=(16, 4))
    )
    p.save(out_path)
    print(f"Mutagenesis profile saved to {out_path}")
