"""
De novo motif and positional feature discovery for translation regulation.

Strategy:
  Phase 1 — Model-intrinsic (fast, no repeated inference):
    1A. Attention positional importance — aggregate across transcripts,
         aligned to CDS start. Single forward pass per sample.
    1B. Input×Gradient saliency — same forward pass as attention.
         d(TE)/d(one_hot), aligned to CDS start.
    1C. AdaLN gene attribution — which genes drive layer-wise modulation.
         Pure weight inspection, zero inference cost.

  Phase 2 — In silico mutagenesis (targeted, cost-aware):
    Only mutate the top-K hotspot positions identified in Phase 1,
    not a full sliding window. This reduces cost from O(L × 4) to O(K × 4).

Key insight for variable-length sequences:
  All position-based metrics are aligned to CDS start (pos 0 = first nt of
  start codon, 1-based in metadata → 0-based in arrays). This makes
  aggregated profiles comparable across transcripts of different lengths.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import defaultdict, Counter
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from eval.calculate_te import calculate_morf_mean_signal


# ============================================================
# Helper: CDS-aligned data extractor for variable-length sequences
# ============================================================
def _extract_sample(dataset, idx):
    """
    Extract a single sample and compute CDS-aligned metadata.
    Returns dict with keys: se, ce, ev, cds_start_0, cds_end_0, L, ct, tid
    All CDS coordinates are 0-based.
    """
    uuid, species, ct, ev, mi, se, ce = dataset[idx]
    se_np = se.cpu().numpy() if torch.is_tensor(se) else np.array(se)
    ce_np = ce.cpu().numpy() if torch.is_tensor(ce) else np.array(ce)
    if torch.is_tensor(ev):
        ev_np = ev.cpu().numpy()
    else:
        ev_np = np.array(ev) if ev is not None else np.zeros(1)

    cds_start = int(mi.get('cds_start_pos', -1)) - 1 if isinstance(mi, dict) else -1
    cds_end = int(mi.get('cds_end_pos', -1)) - 1 if isinstance(mi, dict) else -1

    tid = str(uuid).rsplit('-', 2)[0] if '-' in str(uuid) else str(uuid).split('.')[0]

    return {
        'se': se_np, 'ce': ce_np, 'ev': ev_np,
        'cds_start_0': cds_start, 'cds_end_0': cds_end,
        'L': se_np.shape[0], 'ct': ct, 'tid': tid,
        'valid': cds_start >= 0 and cds_end > cds_start
    }


# ============================================================
# Phase 1A: Attention positional importance (CDS-aligned)
# ============================================================
def extract_attention_positional_importance(
    model, dataset, n_samples=200, max_len=1200, device=None
):
    """
    Forward n_samples, extract per-layer attention weights (keys-received
    per position), and aggregate aligned to CDS start.

    For variable-length sequences, positions beyond the length of a given
    transcript contribute NaN and are excluded from aggregation.

    Returns:
        attn_agg: DataFrame with columns layer, pos_from_cds_start,
                  mean_attn, std_attn, n_contrib
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    n_layers = len(model.encoder.encoder_layers)
    n_heads = model.n_heads
    head_dim = model.encoder.encoder_layers[0].multi_headed_attention.head_dim

    # Per-position accumulators
    accum = defaultdict(lambda: {'sum': 0.0, 'sum_sq': 0.0, 'n': 0})

    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    valid_count = 0

    for idx in tqdm(indices, desc="Attention positional importance"):
        sample = _extract_sample(dataset, idx)
        if not sample['valid']:
            continue
        if sample['L'] > max_len:
            continue

        se = torch.from_numpy(sample['se']).float().unsqueeze(0).to(device)
        ce = torch.from_numpy(sample['ce']).float().unsqueeze(0).to(device)
        ev = torch.from_numpy(sample['ev']).float().unsqueeze(0).to(device)

        L = sample['L']
        cds_start = sample['cds_start_0']

        with torch.no_grad():
            src_embs = model.src_emb(se, ce)
            src_mask = (se[:, :, 0] != 0).to(device)

            species_idx = torch.zeros(1, dtype=torch.long, device=device)
            species_emb = model.species_embedding(species_idx)
            expr_input = torch.cat([ev, species_emb], dim=-1)
            compact_style = model.expr_projector(expr_input)

            src_reps = src_embs

            for layer_idx, encoder_layer in enumerate(model.encoder.encoder_layers):
                sublayer = encoder_layer.sublayers[0]
                style = sublayer.adaLN_modulation(compact_style)
                gamma, beta, alpha = style.chunk(3, dim=-1)
                gamma = gamma.unsqueeze(1)
                beta = beta.unsqueeze(1)
                normed = (1 + gamma) * sublayer.LN(src_reps) + beta

                attn_module = encoder_layer.multi_headed_attention
                bs, Lc, d = normed.shape

                q = attn_module.toqueries(normed).view(bs, Lc, n_heads, head_dim).transpose(1, 2)
                k = attn_module.tokeys(normed).view(bs, Lc, n_heads, head_dim).transpose(1, 2)
                v = attn_module.tovalues(normed).view(bs, Lc, n_heads, head_dim).transpose(1, 2)

                if hasattr(attn_module, 'RoPE'):
                    q = attn_module.RoPE(q)
                    k = attn_module.RoPE(k)

                scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(head_dim)
                mask = src_mask[:, :Lc].unsqueeze(1).unsqueeze(2)
                scores.masked_fill_(~mask, float('-inf'))
                attn_w = torch.softmax(scores, dim=-1)  # (1, H, L, L)

                # Position j receives attention: column sum over query dim
                received = attn_w.sum(dim=2).mean(dim=1)[0].cpu().numpy()  # (L,)

                for pos in range(Lc):
                    rel_pos = pos - cds_start
                    accum[(layer_idx, rel_pos)]['sum'] += float(received[pos])
                    accum[(layer_idx, rel_pos)]['sum_sq'] += float(received[pos]) ** 2
                    accum[(layer_idx, rel_pos)]['n'] += 1

                # Complete forward for next layer
                attn_out = torch.matmul(attn_w, v)
                attn_out = attn_out.transpose(1, 2).reshape(bs, Lc, n_heads * head_dim)
                attn_out = attn_module.unifyheads(attn_out)
                attn_out = attn_module.dropout(attn_out)
                src_reps = src_reps + alpha * sublayer.dropout(attn_out)

                # FFN
                sublayer2 = encoder_layer.sublayers[1]
                style2 = sublayer2.adaLN_modulation(compact_style)
                gamma2, beta2, alpha2 = style2.chunk(3, dim=-1)
                normed2 = (1 + gamma2.unsqueeze(1)) * sublayer2.LN(src_reps) + beta2.unsqueeze(1)
                ffn_out = sublayer2.dropout(encoder_layer.ffn(normed2))
                src_reps = src_reps + alpha2.unsqueeze(1) * ffn_out

        valid_count += 1

    # Build DataFrame
    records = []
    for (layer, pos), v in accum.items():
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

    df = pd.DataFrame(records)
    print(f"Attention aggregated: {len(df)} position-layer pairs from {valid_count} samples")
    return df


# ============================================================
# Phase 1B: Input saliency (CDS-aligned, same forward pass)
# ============================================================
def compute_saliency_profile(model, dataset, n_samples=100, max_len=1200,
                              device=None):
    """
    Compute d(TE)/d(one_hot) for n_samples and aggregate by position
    relative to CDS start. Single forward+backward pass per sample.

    Returns DataFrame: pos_from_cds_start, mean_saliency, std_saliency, n
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    accum = defaultdict(lambda: {'sum': 0.0, 'sum_sq': 0.0, 'n': 0})
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    valid_count = 0

    for idx in tqdm(indices, desc="Input saliency"):
        sample = _extract_sample(dataset, idx)
        if not sample['valid'] or sample['L'] > max_len:
            continue

        se = torch.from_numpy(sample['se']).float().unsqueeze(0).to(device).requires_grad_(True)
        ce = torch.from_numpy(sample['ce']).float().unsqueeze(0).to(device)
        ev = torch.from_numpy(sample['ev']).float().unsqueeze(0).to(device)

        cds_start = sample['cds_start_0']
        cds_end = sample['cds_end_0']
        L = sample['L']

        # Forward — use predict() for clean TE extraction
        with torch.enable_grad():
            out = model.predict(
                seq_batch=se, count_batch=ce, expr_vector=ev,
                head_names=['count'], return_numpy=False,
            )
            pred = out['count']
            if isinstance(pred, dict):
                pred = pred.get('profile', pred)

            # TE = mean frame0 signal in CDS
            if cds_end > cds_start:
                te = pred[0, cds_start:cds_end:3, 0].mean()
            else:
                te = pred[0, :, 0].mean()

        te.backward()
        grad = se.grad[0].detach().cpu().numpy()  # (L, 4)
        sal = np.abs(grad).sum(axis=-1)  # (L,)

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
                'mean_saliency': mean,
                'std_saliency': std,
                'n': v['n'],
            })
    df = pd.DataFrame(records).sort_values('pos_from_cds_start')
    print(f"Saliency aggregated: {len(df)} positions from {valid_count} samples")
    return df


# ============================================================
# Phase 1C: AdaLN gene attribution (zero inference cost)
# ============================================================
def compute_adaLN_gene_attribution(model, gene_names=None, top_k=50):
    """
    Trace which genes drive AdaLN modulation via weight chain:
      gene_i → W_proj1 → d_cell_env → W_proj2 → 32d → W_adaLN → γ/β/α

    Pure weight inspection—no forward pass needed.

    Returns DataFrame: layer_module, gene, gene_idx, score, score_norm
    """
    n_layers = len(model.encoder.encoder_layers)
    d_expr = model.d_expr

    W_proj1 = model.expr_projector[1].weight.detach().cpu().numpy()
    W_proj1_expr = np.abs(W_proj1[:, :d_expr])
    W_proj2 = np.abs(model.expr_projector[3].weight.detach().cpu().numpy())

    all_attr = []
    for layer_idx in range(n_layers):
        for sub_idx, sub_name in [(0, 'attn'), (1, 'ffn')]:
            mod = model.encoder.encoder_layers[layer_idx].sublayers[sub_idx].adaLN_modulation[1]
            W_ada = np.abs(mod.weight.detach().cpu().numpy())
            ada_imp = W_ada.sum(axis=0)

            gene_scores = ada_imp @ W_proj2 @ W_proj1_expr

            top_idx = np.argsort(gene_scores)[::-1][:top_k]
            for gi in top_idx:
                name = gene_names[gi] if gene_names and gi < len(gene_names) else f"GENE_{gi}"
                all_attr.append({
                    'layer': layer_idx,
                    'layer_module': f'L{layer_idx}-{sub_name}',
                    'gene': name,
                    'gene_idx': gi,
                    'score': float(gene_scores[gi]),
                })

    df = pd.DataFrame(all_attr)
    df['score_norm'] = df.groupby('layer_module')['score'].transform(lambda x: x / x.max())
    print(f"Gene attribution: {len(df)} entries, {df['gene'].nunique()} unique genes")
    return df


# ============================================================
# Phase 2: Targeted in silico mutagenesis (only hotspot positions)
# ============================================================
def targeted_mutagenesis(model, dataset, seq_dict, tx_cds,
                          target_positions, n_transcripts=30,
                          cell_type=None, device=None):
    """
    Instead of scanning every position, only mutate the K positions
    identified as hotspots from Phase 1 (attention peaks + saliency peaks).

    For each (transcript, hotspot_position), try all 3 alternative bases
    and measure delta TE.

    target_positions: list of int (relative to CDS start, e.g. [-3, -2, -1, 0, 4, 5, ...])

    Returns DataFrame: tid, pos_from_cds_start, ref_base, mut_base, delta_te
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    nt_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    # Select transcripts that have CDS annotation
    valid_tids = [t for t in seq_dict if t in tx_cds
                  and tx_cds[t].get('cds_start_pos', -1) > 0]
    if len(valid_tids) > n_transcripts:
        selected = np.random.choice(valid_tids, n_transcripts, replace=False)
    else:
        selected = valid_tids

    results = []

    for tid in tqdm(selected, desc="Targeted mutagenesis"):
        cds_info = tx_cds[tid]
        cds_start = cds_info.get('cds_start_pos', -1) - 1  # 0-based
        cds_end = cds_info.get('cds_end_pos', -1) - 1
        if cds_start < 0 or cds_end <= cds_start:
            continue

        seq = seq_dict[tid].upper()
        L = len(seq)

        # Find dataset sample
        sample_idx = None
        for i in range(len(dataset)):
            tid_i = str(dataset[i][0]).rsplit('-', 2)[0]
            if tid_i == tid:
                if cell_type is None or dataset[i][2] == cell_type:
                    sample_idx = i
                    break
        if sample_idx is None:
            continue

        _, species, ct, ev, mi, se_orig, ce_orig = dataset[sample_idx]
        se_ref = se_orig.clone().unsqueeze(0).to(device)
        ce_ref = ce_orig.clone().unsqueeze(0).to(device)
        if torch.is_tensor(ev):
            ev_t = ev.clone().unsqueeze(0).to(device)
        else:
            ev_t = torch.from_numpy(np.array(ev)).float().unsqueeze(0).to(device)

        # Baseline TE
        with torch.no_grad():
            out_ref = model.predict(
                seq_batch=se_ref, count_batch=ce_ref, expr_vector=ev_t,
                head_names=['count'], return_numpy=False,
            )
            pred_ref = out_ref['count']
            if isinstance(pred_ref, dict):
                pred_ref = pred_ref.get('profile', pred_ref)
            te_ref = pred_ref[0, cds_start:cds_end:3, 0].mean().item()

        # Mutate only target positions (relative to CDS start → absolute)
        for rel_pos in target_positions:
            abs_pos = cds_start + rel_pos
            if abs_pos < 0 or abs_pos >= L:
                continue
            orig_base = seq[abs_pos]
            if orig_base not in nt_map:
                continue
            orig_idx = nt_map[orig_base]

            for target_base, target_idx in nt_map.items():
                if target_idx == orig_idx:
                    continue

                se_mut = se_orig.clone().unsqueeze(0).to(device)
                se_mut[0, abs_pos, :] = 0
                se_mut[0, abs_pos, target_idx] = 1.0

                with torch.no_grad():
                    out_mut = model.predict(
                        seq_batch=se_mut, count_batch=ce_ref, expr_vector=ev_t,
                        head_names=['count'], return_numpy=False,
                    )
                    pred_mut = out_mut['count']
                    if isinstance(pred_mut, dict):
                        pred_mut = pred_mut.get('profile', pred_mut)
                    te_mut = pred_mut[0, cds_start:cds_end:3, 0].mean().item()

                results.append({
                    'tid': tid,
                    'pos_from_cds_start': rel_pos,
                    'ref_base': orig_base,
                    'mut_base': target_base,
                    'te_ref': te_ref,
                    'te_mut': te_mut,
                    'delta_te': te_mut - te_ref,
                    'log2_fc': np.log2((te_mut + 1e-8) / (te_ref + 1e-8)),
                })

    df = pd.DataFrame(results)
    n_mutations = len(df)
    # ~n_transcripts * len(target_positions) * 3 mutations
    print(f"Targeted mutagenesis: {n_mutations} mutations "
          f"({len(selected)} transcripts × {len(target_positions)} positions × 3 bases)")
    return df


# ============================================================
# Hotspot extraction from Phase 1 results
# ============================================================
def identify_hotspot_positions(attn_df, saliency_df, n_hotspots=30,
                                layer_range=None):
    """
    Combine attention and saliency signals to identify hotspot positions
    for targeted mutagenesis.

    Strategy: Z-score normalize each metric, average them, take top N.
    """
    # Aggregate attention across layers
    if layer_range is None:
        attn_agg = attn_df.groupby('pos_from_cds_start')['mean_attn'].mean().reset_index()
    else:
        mask = attn_df['layer'].between(*layer_range)
        attn_agg = attn_df[mask].groupby('pos_from_cds_start')['mean_attn'].mean().reset_index()

    # Merge with saliency
    merged = attn_agg.merge(saliency_df, on='pos_from_cds_start', how='outer').fillna(0)

    # Z-score
    merged['attn_z'] = (merged['mean_attn'] - merged['mean_attn'].mean()) / (merged['mean_attn'].std() + 1e-8)
    merged['sal_z'] = (merged['mean_saliency'] - merged['mean_saliency'].mean()) / (merged['mean_saliency'].std() + 1e-8)
    merged['combined_score'] = (merged['attn_z'] + merged['sal_z']) / 2

    hotspots = merged.nlargest(n_hotspots, 'combined_score')
    positions = sorted(hotspots['pos_from_cds_start'].astype(int).tolist())

    print(f"Identified {len(positions)} hotspot positions: {positions}")
    return positions, hotspots


# ============================================================
# Sequence context extraction at hotspots
# ============================================================
def extract_hotspot_contexts(seq_dict, tx_cds, hotspot_positions,
                              context_radius=20, max_seqs=200):
    """
    Extract sequence contexts around hotspot positions for MEME input.
    Returns {rel_pos: [sequence_contexts]}.
    """
    contexts = defaultdict(list)
    tids = [t for t in seq_dict if t in tx_cds
            and tx_cds[t].get('cds_start_pos', -1) > 0]

    for tid in tids[:max_seqs]:
        cds_start = tx_cds[tid].get('cds_start_pos', -1) - 1
        seq = seq_dict[tid].upper()

        for rel_pos in hotspot_positions:
            abs_pos = cds_start + rel_pos
            if 0 <= abs_pos < len(seq):
                ctx_start = max(0, abs_pos - context_radius)
                ctx_end = min(len(seq), abs_pos + context_radius + 1)
                ctx = seq[ctx_start:ctx_end]
                if 'N' not in ctx:
                    contexts[rel_pos].append(ctx)

    return dict(contexts)


# ============================================================
# Aggregate mutagenesis by position
# ============================================================
def aggregate_mutagenesis(mut_df):
    """Aggregate targeted mutagenesis results by position."""
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
