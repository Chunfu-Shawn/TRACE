"""
De novo motif and positional feature discovery for translation regulation.

All position metrics are mapped to a Metagene Coordinate System:
  - 5' UTR: True nucleotide distance from CDS start (< 0)
  - CDS: Length-proportionally mapped to a fixed length (e.g., 900 nt), strictly preserving reading frame.
  - 3' UTR: True nucleotide distance from CDS stop (>= fixed_cds_len)
"""

import os, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import defaultdict, Counter
from tqdm import tqdm
import warnings
from eval.calculate_te import *
import logomaker
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# ============================================================
# Global Parameter
# ============================================================
FIXED_CDS_LEN = 600  # The normalized length for all CDS regions (must be a multiple of 3)

# ============================================================
# Metagene Mapping Utilities
# ============================================================
def _map_to_metagene(pos, cds_start, cds_end, fixed_cds_len=FIXED_CDS_LEN):
    """
    Map absolute physical position to a unified metagene coordinate (x_pos),
    preserving exact nucleotide distance in UTRs, and proportional length in CDS.
    Strictly preserves the 0/1/2 reading frame periodicity.
    """
    rel_start = pos - cds_start
    rel_stop = pos - cds_end

    if rel_start < 0: # 5' UTR
        x_pos = rel_start
    elif rel_stop >= 0: # 3' UTR
        x_pos = fixed_cds_len + rel_stop
    else: # CDS Internal proportional sampling
        cds_len = cds_end - cds_start
        codon_idx = rel_start // 3
        frame = rel_start % 3
        total_codons = cds_len // 3
        target_codons = fixed_cds_len // 3
        
        if total_codons > 0:
            # Proportional mapping at codon level to preserve frame
            mapped_codon = int(np.round((codon_idx / total_codons) * target_codons))
            mapped_codon = min(mapped_codon, target_codons - 1)
        else:
            mapped_codon = 0
            
        x_pos = mapped_codon * 3 + frame

    return x_pos, rel_start, rel_stop

def _inverse_metagene(x_pos, cds_start, cds_end, fixed_cds_len=FIXED_CDS_LEN):
    """
    Inverse map a unified metagene coordinate back to the absolute physical position 
    for a specific transcript (used for targeted mutagenesis).
    """
    cds_len = cds_end - cds_start
    
    if x_pos < 0:
        rel_start = x_pos
    elif x_pos >= fixed_cds_len:
        rel_start = cds_len + (x_pos - fixed_cds_len)
    else:
        target_codon = x_pos // 3
        frame = x_pos % 3
        total_codons = cds_len // 3
        target_codons = fixed_cds_len // 3
        
        if target_codons > 0:
            codon_idx = int(np.round((target_codon / target_codons) * total_codons))
            codon_idx = min(codon_idx, total_codons - 1)
        else:
            codon_idx = 0
        rel_start = codon_idx * 3 + frame
        
    return cds_start + rel_start

# ============================================================
# Helper functions
# ============================================================
def _unwrap(model):
    return model.module if hasattr(model, 'module') else model

def _extract_sample(dataset, idx):
    uuid, species, ct, ev, mi, se, ce = dataset[idx]

    se_np = se.cpu().numpy() if torch.is_tensor(se) else np.array(se)
    ce_np = ce.cpu().numpy() if torch.is_tensor(ce) else np.array(ce)

    if torch.is_tensor(ev):
        ev_np = ev.cpu().numpy()
    else:
        ev_np = np.array(ev) if ev is not None else None
    if ev_np is not None and ev_np.ndim == 1 and ev_np.shape[0] == 0:
        ev_np = None

    cds_start = int(mi.get('cds_start_pos', -1)) - 1 if isinstance(mi, dict) else -1
    cds_end = int(mi.get('cds_end_pos', -1)) if isinstance(mi, dict) else -1
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
def extract_attention_positional_importance(model, dataset, n_samples=200, min_len=500, max_len=1200, device=None):
    raw = _unwrap(model)
    if device is None:
        device = next(raw.parameters()).device
    raw.eval()

    n_layers = len(raw.encoder.encoder_layers)
    n_heads = raw.n_heads
    head_dim = raw.encoder.encoder_layers[0].multi_headed_attention.head_dim

    # Now using a unified metagene accumulator
    accum = defaultdict(lambda: {'sum': 0.0, 'sum_sq': 0.0, 'n': 0, 'rel_start_sum': 0.0, 'rel_stop_sum': 0.0})
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    valid_count = 0

    for idx in tqdm(indices, desc="Attention positional importance"):
        s = _extract_sample(dataset, idx)
        if not s['valid'] or s['L'] > max_len or s['L'] < min_len:
            continue

        se = torch.from_numpy(s['se']).float().unsqueeze(0).to(device)
        ce = torch.from_numpy(s['ce']).float().unsqueeze(0).to(device)
        ev = torch.from_numpy(s['ev']).float().unsqueeze(0).to(device) if s['ev'] is not None and len(s['ev']) > 0 else None
        
        cds_start, cds_end = s['cds_start_0'], s['cds_end_0']

        with torch.no_grad():
            resolved_expr = raw._resolve_expr_vector(cell_type=s['ct'], expr_vector=ev, batch_size=1).to(device)
            species_idx = raw._normalize_species(s['species'], 1).to(device)
            species_emb = raw.species_embedding(species_idx)
            combined_env = torch.cat([resolved_expr, species_emb], dim=-1)
            compact_style = raw.expr_projector(combined_env)

            src_reps = raw.src_emb(se, ce)
            src_mask = (se[:, :, 0] != 0).to(device)

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
                    # Proportional mapping for unification
                    x_pos, rel_start, rel_stop = _map_to_metagene(pos, cds_start, cds_end, FIXED_CDS_LEN)
                    
                    key = (layer_idx, x_pos)
                    accum[key]['sum'] += float(received[pos])
                    accum[key]['sum_sq'] += float(received[pos]) ** 2
                    accum[key]['n'] += 1
                    accum[key]['rel_start_sum'] += rel_start
                    accum[key]['rel_stop_sum'] += rel_stop

                attn_out = torch.matmul(attn_w, v)
                attn_out = attn_out.transpose(1, 2).reshape(bs_, Lc, n_heads * head_dim)
                attn_out = attn_mod.unifyheads(attn_out)
                if hasattr(attn_mod, 'dropout'):
                    attn_out = attn_mod.dropout(attn_out)
                src_reps = src_reps + alpha.unsqueeze(1) * sub.dropout(attn_out)

                sub2 = enc_layer.sublayers[1]
                style2 = sub2.adaLN_modulation(compact_style)
                gamma2, beta2, alpha2 = style2.chunk(3, dim=-1)
                normed2 = (1 + gamma2.unsqueeze(1)) * sub2.LN(src_reps) + beta2.unsqueeze(1)
                ffn_out = sub2.dropout(enc_layer.ffn(normed2))
                src_reps = src_reps + alpha2.unsqueeze(1) * ffn_out

        valid_count += 1

    records = []
    for (layer, x_pos), v in accum.items():
        if v['n'] >= 5:
            mean = v['sum'] / v['n']
            std = np.sqrt(max(0, v['sum_sq'] / v['n'] - mean ** 2))
            records.append({
                'layer': layer,
                'x_pos': x_pos,
                'mean_attn': mean,
                'std_attn': std,
                'pos_from_cds_start': v['rel_start_sum'] / v['n'], # Averaged physical relative start
                'pos_from_cds_stop': v['rel_stop_sum'] / v['n'],   # Averaged physical relative stop
                'n_contrib': v['n'],
            })

    df = pd.DataFrame(records).sort_values(['layer', 'x_pos'])
    print(f"Attention aggregated: {len(df)} metagene positions from {valid_count} samples.")
    return df

# ============================================================
# Phase 1B: Input saliency (Modified for Whole Profile Shape)
# ============================================================
def compute_saliency_profile(model, dataset, n_samples=100, max_len=1200, device=None):
    raw = _unwrap(model)
    if device is None:
        device = next(raw.parameters()).device
    raw.eval()

    accum = defaultdict(lambda: {'sum': 0.0, 'sum_sq': 0.0, 'n': 0, 'rel_start_sum': 0.0, 'rel_stop_sum': 0.0})
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    valid_count = 0

    for idx in tqdm(indices, desc="Input saliency (Whole Profile)"):
        s = _extract_sample(dataset, idx)
        if not s['valid'] or s['L'] > max_len:
            continue
            
        se = torch.from_numpy(s['se']).float().unsqueeze(0).to(device).requires_grad_(True)
        ce = torch.from_numpy(s['ce']).float().unsqueeze(0).to(device)
        ev = torch.from_numpy(s['ev']).float().unsqueeze(0).to(device) if s['ev'] is not None and len(s['ev']) > 0 else None
        
        L, cds_start, cds_end = s['L'], s['cds_start_0'], s['cds_end_0']

        raw.eval()
        with torch.enable_grad():
            resolved_expr = raw._resolve_expr_vector(cell_type=s['ct'], expr_vector=ev, batch_size=1).to(device)
            species_idx = raw._normalize_species(s['species'], 1).to(device)

            out = raw.forward(
                seq_batch=se, count_batch=ce,
                expr_vector=resolved_expr, species=species_idx,
                head_names=['count'],
            )
            pred = out['count']
            if isinstance(pred, dict):
                pred = pred.get('profile', pred)
            
            # [Mod]: 捕捉整体 Profile 的形状和振幅变化，而不仅仅是均值
            # 求平方和 (L2 Norm squared) 可以确保波峰的正向和负向扰动都不会被抵消
            #profile_tensor = pred[0, :, 0]
            #profile_loss = (profile_tensor ** 2).sum()
            te = pred[0, cds_start:cds_end, 0].mean()

        te.backward()
        grad = se.grad[0].detach().cpu().numpy()
        sal = np.abs(grad).sum(axis=-1)

        for pos in range(L):
            x_pos, rel_start, rel_stop = _map_to_metagene(pos, cds_start, cds_end, FIXED_CDS_LEN)
            accum[x_pos]['sum'] += float(sal[pos])
            accum[x_pos]['sum_sq'] += float(sal[pos]) ** 2
            accum[x_pos]['n'] += 1
            accum[x_pos]['rel_start_sum'] += rel_start
            accum[x_pos]['rel_stop_sum'] += rel_stop

        se.grad = None
        valid_count += 1

    records = []
    for x_pos, v in accum.items():
        if v['n'] >= 5:
            mean = v['sum'] / v['n']
            std = np.sqrt(max(0, v['sum_sq'] / v['n'] - mean ** 2))
            records.append({
                'x_pos': x_pos,
                'mean_saliency': mean, 'std_saliency': std,
                'pos_from_cds_start': v['rel_start_sum'] / v['n'],
                'pos_from_cds_stop': v['rel_stop_sum'] / v['n'],
                'n_contrib': v['n'],
            })

    df = pd.DataFrame(records).sort_values('x_pos')
    print(f"Saliency aggregated: {len(df)} metagene positions from {valid_count} samples")
    return df


def run_differential_saliency(model, dataset, cell_type_A, cell_type_B, max_len=1200, n_samples=500):
    # 1. 从数据集中筛出特定细胞系的样本索引
    indices_A = [i for i, d in enumerate(dataset) if d[2] == cell_type_A]
    indices_B = [i for i, d in enumerate(dataset) if d[2] == cell_type_B]
    
    # 构建临时的小型 Dataset 子集 (方便传给 compute_saliency_profile)
    from torch.utils.data import Subset
    subset_A = Subset(dataset, indices_A)
    subset_B = Subset(dataset, indices_B)

    # 2. 分别计算
    print(f"Running Saliency for {cell_type_A}...")
    sal_A = compute_saliency_profile(model, subset_A, n_samples=min(n_samples, len(indices_A)), max_len=max_len, device=device)
    
    print(f"Running Saliency for {cell_type_B}...")
    sal_B = compute_saliency_profile(model, subset_B, n_samples=min(n_samples, len(indices_B)), max_len=max_len, device=device)
    
    # 3. 计算差分 (Delta Saliency)
    # 假设我们只关心在 A 中起作用而在 B 中不起作用的位点
    merged = pd.merge(sal_A[['x_pos', 'mean_saliency']], sal_B[['x_pos', 'mean_saliency']], on='x_pos', suffixes=('_A', '_B'))
    merged['delta_saliency'] = merged['mean_saliency_A'] - merged['mean_saliency_B']
    
    # 取 Delta 最大的前 50 个 Metagene 坐标
    top_diff_hotspots = merged.nlargest(50, 'delta_saliency')['x_pos'].tolist()
    print(f"Top Differential Hotspots (Active in {cell_type_A}, Silent in {cell_type_B}): {top_diff_hotspots}")
    
    return merged, top_diff_hotspots

# ============================================================
# Phase 1C: AdaLN gene attribution (Unchanged logic, just compacted)
# ============================================================
def _load_gene_names(gene_order_path=None, gene_annot_path="/home/user/data3/rbase/genome_ref/Homo_sapiens/hg38/ens_genes_v112.txt"):
    import os as _os
    if gene_order_path is None:
        gene_order_path = "/home/user/data3/rbase/translation_model/models/src/config/global_anchor_gene_order.txt"
    with open(gene_order_path) as f:
        ensg_list = [line.strip() for line in f if line.strip()]

    ensg2name = {}
    if gene_annot_path is not None:
        with open(gene_annot_path) as f:
            header = f.readline().strip().split('\t')
            gid_col = header.index('Gene stable ID') if 'Gene stable ID' in header else 0
            gname_col = header.index('Gene name') if 'Gene name' in header else 2
            for line in f:
                cols = line.strip().split('\t')
                if len(cols) > max(gid_col, gname_col):
                    ensg2name[cols[gid_col]] = cols[gname_col]

    gene_names = [ensg2name.get(e, e) for e in ensg_list]
    return gene_names


def compute_adaLN_gene_attribution(model, gene_names=None, top_k=50, gene_annot_path=None):
    raw = _unwrap(model)
    n_layers, d_expr = len(raw.encoder.encoder_layers), raw.d_expr
    gene_names = gene_names or _load_gene_names()

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
                    'layer': layer_idx, 'layer_module': f'L{layer_idx}-{sub_name}',
                    'gene': name, 'gene_idx': gi, 'score': float(gene_scores[gi]),
                })

    df = pd.DataFrame(all_attr)
    df['score_norm'] = df.groupby('layer_module')['score'].transform(lambda x: x / x.max())
    return df

    
# ============================================================
# Plotting utilities — Single continuous axis, per-layer color
# ============================================================

def _assign_frame_colors(df):
    """
    Since the metagene coordinate x_pos intrinsically preserves the frame,
    we can safely calculate frame directly from x_pos.
    Frame 0: Red (#E41A1C), Frame 1: Blue (#377EB8), Frame 2: Gray (gray)
    """
    df['frame'] = df['x_pos'].astype(int) % 3
    # [Mod]: Frame 2 updated to 'gray' based on user preference
    color_map = {0: '#E41A1C', 1: '#377EB8', 2: 'gray'}
    df['frame_color'] = df['frame'].map(color_map)
    df['Frame'] = df['frame'].map({0: 'Frame 0', 1: 'Frame 1', 2: 'Frame 2'})
    df['Frame'] = pd.Categorical(df['Frame'], categories=['Frame 0', 'Frame 1', 'Frame 2'])
    return df

def _cds_rect_data():
    """Build geom_rect data for a single continuous CDS shading."""
    return pd.DataFrame({
        'xmin': [0], 
        'xmax': [FIXED_CDS_LEN], 
        'ymin': [-float('inf')], 
        'ymax': [float('inf')], 
        'fill': ['lightgray']
    })


def plot_attention_profile(attn_df, out_path="attention_profile.pdf", up_len=300, down_len=300, 
                           color_by_frame=True, xlim=None, show_xaxis=False, show_cds=True, 
                           weight=6, height=5):
    
    from plotnine import (ggplot, aes, geom_line, geom_rect,
                          labs, theme, facet_grid, scale_color_manual, scale_fill_identity,
                          element_text, theme_classic, element_blank, element_line)
    import pandas as pd
    import numpy as np
    
    # 1. Bounds filtering (support explicit xlim for zooming in)
    if xlim is not None:
        df_plot = attn_df[(attn_df['x_pos'] >= xlim[0]) & (attn_df['x_pos'] <= xlim[1])].copy()
    else:
        df_plot = attn_df[(attn_df['x_pos'] >= -up_len) & (attn_df['x_pos'] <= FIXED_CDS_LEN + down_len - 1)].copy()
        
    df_plot['layer'] = df_plot['layer'].astype(int)

    # 2. Aggregation logic based on whether we group by Frame
    if color_by_frame:
        df_plot = _assign_frame_colors(df_plot)
        group_cols = ['layer', 'x_pos', 'Frame']
    else:
        group_cols = ['layer', 'x_pos']

    df_plot = df_plot.groupby(group_cols, as_index=False, observed=True)[['mean_attn']].mean().dropna(subset=['mean_attn'])
    df_plot['log2_mean_attn'] = np.log2(df_plot['mean_attn'] + 1)

    base_out = out_path.replace('.pdf', '')
    rect_cds = _cds_rect_data()
    
    # Dynamic axis styling based on show_xaxis
    x_axis_text = element_text() if show_xaxis else element_blank()
    x_axis_ticks = element_line() if show_xaxis else element_blank()
    x_axis_title = element_text() if show_xaxis else element_blank()
    x_label_str = 'Metagene Position (x_pos)' if show_xaxis else ''

    # ==================================
    # Combined Plot
    # ==================================
    if color_by_frame:
        df_combined = df_plot.groupby(['x_pos', 'Frame'], as_index=False, observed=True)[['mean_attn']].mean()
    else:
        df_combined = df_plot.groupby(['x_pos'], as_index=False, observed=True)[['mean_attn']].mean()
        
    df_combined['log2_mean_attn'] = np.log2(df_combined['mean_attn'] + 1)

    p_comb = (
        ggplot(df_combined, aes(x='x_pos', y='log2_mean_attn'))
        + scale_fill_identity()
    )

    if show_cds:
        p_comb += geom_rect(data=rect_cds, mapping=aes(xmin='xmin', xmax='xmax', ymin='ymin', ymax='ymax', fill='fill'), alpha=0.3, inherit_aes=False, show_legend=False)

    if color_by_frame:
        frame_palette = {'Frame 0': '#E41A1C', 'Frame 1': '#377EB8', 'Frame 2': 'gray'}
        # [修改]: 替换为 geom_smooth 进行平滑拟合，se=False 代表不画置信区间阴影
        p_comb += geom_line(aes(color='Frame', group='Frame'), size=0.8, alpha=0.6) 
        p_comb += scale_color_manual(values=frame_palette)
    else:
        # [修改]: Monocolor 聚合视图的平滑拟合
        p_comb += geom_line(size=0.8, alpha=0.6, color='#333333') 

    p_comb += labs(x=x_label_str, y='log2(Mean attention + 1)') 
    p_comb += theme_classic()
    p_comb += theme(axis_text_x=x_axis_text, axis_ticks_major_x=x_axis_ticks, axis_title_x=x_axis_title, figure_size=(weight, height))
    
    p_comb.save(f"{base_out}.combined.pdf")
    print(f"Combined attention profile saved to {base_out}.combined.pdf")

    # ==================================
    # Per-layer Plot
    # ==================================
    n_layers = df_plot['layer'].nunique()
    df_plot['Layer'] = pd.Categorical([f'L{li}' for li in df_plot['layer']], categories=[f'L{i}' for i in range(n_layers)])

    rect_per_layer = pd.DataFrame({
        'Layer': pd.Categorical([f'L{i}' for i in range(n_layers)], categories=[f'L{i}' for i in range(n_layers)]),
        'xmin': [0] * n_layers, 'xmax': [FIXED_CDS_LEN] * n_layers,
        'ymin': [-float('inf')] * n_layers, 'ymax': [float('inf')] * n_layers,
        'fill': ['lightgray'] * n_layers
    })

    p_layers = (
        ggplot(df_plot, aes(x='x_pos', y='log2_mean_attn'))
        + scale_fill_identity()
    )

    if show_cds:
        p_layers += geom_rect(data=rect_per_layer, mapping=aes(xmin='xmin', xmax='xmax', ymin='ymin', ymax='ymax', fill='fill'), alpha=0.3, inherit_aes=False, show_legend=False)
    
    if color_by_frame:
        # [修改]: 分层图平滑拟合
        p_layers += geom_line(aes(color='Frame', group='Frame'), size=0.6, alpha=0.6)
        p_layers += scale_color_manual(values=frame_palette)
    else:
        # [修改]: 分层图平滑拟合
        p_layers += geom_line(size=0.6, alpha=0.6, color='#333333')

    p_layers += facet_grid('Layer ~ .', scales='free_y')
    p_layers += labs(x=x_label_str, y='log2(Mean attention + 1)')
    p_layers += theme_classic()
    p_layers += theme(axis_text_x=x_axis_text, axis_ticks_major_x=x_axis_ticks, axis_title_x=x_axis_title, 
                      strip_background=element_blank(), strip_text=element_text(size=12), figure_size=(weight, height*3))
    
    p_layers.save(f"{base_out}.per_layer.pdf")


def plot_regional_attention_dynamics(attn_df, out_path="regional_attention_dynamics.pdf", up_len=300, down_len=300):
    """
    Plots the layer-by-layer dynamic shifts in attention across 5 specific regions:
    5' UTR, CDS (Frame 0), CDS (Frame 1), CDS (Frame 2), and 3' UTR.
    Produces both an absolute mean attention line plot and a 100% relative proportion bar chart.
    """
    from plotnine import (ggplot, aes, geom_line, geom_point, geom_col, position_stack,
                          labs, theme_classic, scale_color_manual, scale_fill_manual, 
                          scale_x_continuous, theme, element_text)
    import pandas as pd
    import numpy as np

    # 1. Filter sequences based on the upstream/downstream boundaries
    df = attn_df[(attn_df['x_pos'] >= -up_len) & (attn_df['x_pos'] <= FIXED_CDS_LEN + down_len - 1)].copy()

    # 2. Annotate the 5 regions
    conditions = [
        df['x_pos'] < 0,
        (df['x_pos'] >= 0) & (df['x_pos'] < FIXED_CDS_LEN) & (df['x_pos'] % 3 == 0),
        (df['x_pos'] >= 0) & (df['x_pos'] < FIXED_CDS_LEN) & (df['x_pos'] % 3 == 1),
        (df['x_pos'] >= 0) & (df['x_pos'] < FIXED_CDS_LEN) & (df['x_pos'] % 3 == 2),
        df['x_pos'] >= FIXED_CDS_LEN
    ]
    choices = ["5' UTR", "CDS (Frame 0)", "CDS (Frame 1)", "CDS (Frame 2)", "3' UTR"]
    
    # [Fix]: Added explicit string default to satisfy Numpy's strict type promotion
    df['Region'] = np.select(conditions, choices, default="Unknown")

    # Convert to Categorical to maintain a strict legend order
    region_order = ["5' UTR", "CDS (Frame 0)", "CDS (Frame 1)", "CDS (Frame 2)", "3' UTR"]
    df['Region'] = pd.Categorical(df['Region'], categories=region_order)

    # 3. Aggregate Mean Attention per Region per Layer
    # Using 'mean' perfectly balances the varying lengths of UTRs and CDS subsets
    agg_df = df.groupby(['layer', 'Region'], as_index=False, observed=True)[['mean_attn']].mean()
    
    # Scale up by 1000 for cleaner Y-axis numbers in the absolute plot
    agg_df['mean_attn_scaled'] = agg_df['mean_attn']

    # Define color map to perfectly match your previous plots
    color_map = {
        "5' UTR": "#FF7F00",       # Orange for 5' UTR
        "CDS (Frame 0)": "#E41A1C", # Red
        "CDS (Frame 1)": "#377EB8", # Blue
        "CDS (Frame 2)": "gray",    # Gray
        "3' UTR": "#984EA3"         # Purple for 3' UTR
    }

    base_out = out_path.replace('.pdf', '')
    max_layer = int(agg_df['layer'].max())

    # ============================================================
    # Plot 1: Absolute Mean Attention per Nucleotide (Line Plot)
    # ============================================================
    p_line = (
        ggplot(agg_df, aes(x='layer', y='mean_attn_scaled', color='Region', group='Region'))
        + geom_line(size=1.2, alpha=0.9)
        + geom_point(size=3)
        + scale_color_manual(values=color_map)
        + scale_x_continuous(breaks=range(0, max_layer + 1))
        + labs(x='Transformer Layer', 
               y='Mean Attention per nt', 
               title='Layer-wise Absolute Attention Dynamics')
        + theme_classic()
        + theme(figure_size=(7, 5),
                axis_text=element_text(size=10),
                title=element_text(size=12, face="bold"))
    )
    p_line.save(f"{base_out}.line.pdf")
    print(f"Regional dynamics (Line) saved to {base_out}.line.pdf")

    # ============================================================
    # Plot 2: Relative Contribution Proportion (100% Stacked Bar)
    # ============================================================
    # Calculate relative proportion for each layer
    layer_sums = agg_df.groupby('layer')['mean_attn'].transform('sum')
    agg_df['relative_prop'] = agg_df['mean_attn'] / layer_sums

    p_bar = (
        ggplot(agg_df, aes(x='layer', y='relative_prop', fill='Region'))
        # Reverse the stack so 5'UTR is at the bottom, matching 5'->3' direction intuitively
        + geom_col(position=position_stack(reverse=True), color='white', size=0.2)
        + scale_fill_manual(values=color_map)
        + scale_x_continuous(breaks=range(0, max_layer + 1))
        + labs(x='Transformer Layer', 
               y='Relative Regional Contribution (100%)', 
               title='Layer-wise Relative Attention Shift')
        + theme_classic()
        + theme(figure_size=(7, 5),
                axis_text=element_text(size=10),
                title=element_text(size=12, face="bold"))
    )
    p_bar.save(f"{base_out}.proportion.pdf")
    print(f"Regional dynamics (Proportion Bar) saved to {base_out}.proportion.pdf")

def plot_saliency_profile(sal_df, out_path="saliency_profile.pdf", up_len=300, down_len=300):
    from plotnine import (ggplot, aes, geom_point, geom_line, geom_rect,
                          labs, theme_classic, theme, scale_color_manual, scale_fill_identity, element_blank)

    df_plot = sal_df[(sal_df['x_pos'] >= -up_len) & (sal_df['x_pos'] <= FIXED_CDS_LEN + down_len - 1)].copy()
    df_plot = _assign_frame_colors(df_plot)
    
    df_plot = df_plot.groupby(['x_pos', 'Frame'], as_index=False, observed=True)[['mean_saliency']].mean().dropna(subset=['mean_saliency'])
    df_plot['log2_saliency'] = np.log2(df_plot['mean_saliency'] + 1)
    
    frame_palette = {'Frame 0': '#E41A1C', 'Frame 1': '#377EB8', 'Frame 2': 'gray'}
    rect_cds = _cds_rect_data()

    p = (
        ggplot(df_plot, aes(x='x_pos', y='log2_saliency'))
        + geom_rect(data=rect_cds, mapping=aes(xmin='xmin', xmax='xmax', ymin='ymin', ymax='ymax', fill='fill'), alpha=0.3, inherit_aes=False, show_legend=False)
        + scale_fill_identity()
        + geom_line(aes(color='Frame', group='Frame'), size=0.8, alpha=0.9)
        + geom_point(aes(color='Frame', group='Frame'), size=0.3, alpha=0.3)
        + scale_color_manual(values=frame_palette)
        + labs(x='', y='log2(Mean |d(profile)/d(base)| + 1)')
        + theme_classic()
        + theme(axis_text_x=element_blank(), axis_ticks_major_x=element_blank(), axis_title_x=element_blank(), figure_size=(6, 4))
    )
    p.save(out_path)
    print(f"Saliency profile saved to {out_path}")


def plot_mutagenesis_profile(pos_agg, out_path="mutagenesis_profile.pdf", up_len=300, down_len=300):
    from plotnine import (ggplot, aes, geom_point, geom_line, geom_rect,
                          labs, theme_classic, theme, scale_color_manual, scale_fill_identity, element_blank)

    df_plot = pos_agg[(pos_agg['x_pos'] >= -up_len) & (pos_agg['x_pos'] <= FIXED_CDS_LEN + down_len - 1)].copy()
    df_plot = _assign_frame_colors(df_plot)
    
    df_plot = df_plot.groupby(['x_pos', 'Frame'], as_index=False, observed=True)[['mean_abs_delta']].mean().dropna(subset=['mean_abs_delta'])
    df_plot['log2_abs_delta'] = np.log2(df_plot['mean_abs_delta'] + 1)
    
    frame_palette = {'Frame 0': '#E41A1C', 'Frame 1': '#377EB8', 'Frame 2': 'gray'}
    rect_cds = _cds_rect_data()

    p = (
        ggplot(df_plot, aes(x='x_pos', y='log2_abs_delta'))
        + geom_rect(data=rect_cds, mapping=aes(xmin='xmin', xmax='xmax', ymin='ymin', ymax='ymax', fill='fill'), alpha=0.3, inherit_aes=False, show_legend=False)
        + scale_fill_identity()
        + geom_line(aes(color='Frame', group='Frame'), size=0.8, alpha=0.9)
        + geom_point(aes(color='Frame', group='Frame'), size=0.3, alpha=0.3)
        + scale_color_manual(values=frame_palette)
        + labs(x='', y='log2(Mean |Delta profile| + 1)')
        + theme_classic()
        + theme(axis_text_x=element_blank(), axis_ticks_major_x=element_blank(), axis_title_x=element_blank(), figure_size=(6, 4))
    )
    p.save(out_path)
    print(f"Mutagenesis profile saved to {out_path}")


def plot_gene_attribution(attr_df, out_path="gene_attribution.pdf", top_n=30):
    """
    Plots a heatmap of the top contributing environmental genes across different model layers.
    Args:
        attr_df: DataFrame generated by compute_adaLN_gene_attribution.
        top_n: Number of globally top contributing genes to display.
    """
    from plotnine import (ggplot, aes, geom_tile, scale_fill_cmap, labs, 
                          theme_classic, theme, element_text, element_blank)
    import pandas as pd

    # 1. 寻找全局贡献最大的 Top N 基因
    gene_totals = attr_df.groupby('gene')['score'].sum().reset_index()
    top_genes = gene_totals.nlargest(top_n, 'score')['gene'].tolist()

    # 2. 筛选数据
    df_plot = attr_df[attr_df['gene'].isin(top_genes)].copy()

    # 3. 设置分类排序逻辑以保证画图顺序美观
    # Y轴：总分越高的基因排在越上面
    df_plot['gene'] = pd.Categorical(df_plot['gene'], categories=top_genes[::-1])
    
    # X轴：按照网络深度的顺序 (L0-attn, L0-ffn, L1-attn, ...)
    layer_order = []
    for l in sorted(attr_df['layer'].unique()):
        layer_order.extend([f"L{l}-attn", f"L{l}-ffn"])
    # 只保留实际存在的列
    layer_order = [l for l in layer_order if l in attr_df['layer_module'].unique()]
    df_plot['layer_module'] = pd.Categorical(df_plot['layer_module'], categories=layer_order)

    p = (
        ggplot(df_plot, aes(x='layer_module', y='gene', fill='score_norm'))
        + geom_tile(color='white', size=0.2)
        # 使用炽热色系 (YlOrRd) 呈现基因调控的重要程度
        + scale_fill_cmap(cmap_name='YlOrRd') 
        + labs(
            x='Model Layer & Module (AdaLN)', 
            y='Environmental Gene (Cell Context)', 
            fill='Normalized\nAttribution'
        )
        + theme_classic()
        + theme(
            axis_text_x=element_text(rotation=45, hjust=1),
            axis_text_y=element_text(size=10),
            # 动态调整图片高度以适配基因数量
            figure_size=(max(8, len(layer_order)*0.4), max(5, top_n*0.25))
        )
    )
    p.save(out_path)
    print(f"Gene attribution heatmap saved to {out_path}")



def compute_ovr_differential_saliency(model, dataset, target_cell_type, all_cell_types, 
                                      n_samples_per_group=300, max_len=2000, device=None):
    """
    计算特定细胞系 (Target) 相较于其他所有细胞系 (Rest) 的差分 Saliency。
    """
    from torch.utils.data import Subset
    
    # 1. 构建 Target 和 Rest 的子集
    idx_target = [i for i, d in enumerate(dataset) if d[2] == target_cell_type]
    idx_rest = [i for i, d in enumerate(dataset) if d[2] in all_cell_types and d[2] != target_cell_type]
    
    if len(idx_target) == 0 or len(idx_rest) == 0:
        print(f"Skipping {target_cell_type} due to insufficient samples.")
        return None, []
        
    subset_target = Subset(dataset, idx_target)
    subset_rest = Subset(dataset, idx_rest)
    
    print(f"\n--- OvR Analysis for {target_cell_type} vs Rest ---")
    
    # 2. 分别计算 Saliency
    sal_target = compute_saliency_profile(
        model, subset_target, 
        n_samples=min(n_samples_per_group, len(idx_target)), 
        max_len=max_len, device=device
    )
    
    sal_rest = compute_saliency_profile(
        model, subset_rest, 
        n_samples=min(n_samples_per_group, len(idx_rest)), 
        max_len=max_len, device=device
    )
    
    # 3. 合并计算差分
    merged = pd.merge(sal_target[['x_pos', 'mean_saliency']], 
                      sal_rest[['x_pos', 'mean_saliency']], 
                      on='x_pos', suffixes=('_Target', '_Rest'))
    
    merged['delta_saliency'] = merged['mean_saliency_Target'] - merged['mean_saliency_Rest']
    
    # 提取差异最大的前 30 个宏观热点
    top_diff_hotspots = merged.nlargest(30, 'delta_saliency')['x_pos'].tolist()
    
    return merged, top_diff_hotspots


def extract_context_with_saliency_filter(model, dataset, seq_dict, tx_cds, 
                                         target_cell_type, x_pos_hotspots, 
                                         context_radius=15, max_seqs=300, device=None):
    """
    微观切片：拿着宏观热点去真实转录本上切出短序列。
    加入 Saliency 过滤器：只保留那些在该物理位点上确实有显著响应的转录本片段。
    """
    raw = _unwrap(model) if hasattr(model, 'module') else model
    
    # [关键修复 1]: 如果调用时未传入 device，自动从模型权重上继承 device
    if device is None:
        device = next(raw.parameters()).device
        
    raw.eval()
    
    # 找到目标细胞系的所有样本
    target_samples = [(i, d) for i, d in enumerate(dataset) if d[2] == target_cell_type]
    selected_samples = np.random.choice(len(target_samples), min(max_seqs, len(target_samples)), replace=False)
    
    valid_contexts = []
    
    print(f"Extracting physical sequence contexts for {target_cell_type}...")
    for idx in tqdm(selected_samples, desc="Micro-slicing"):
        real_idx, d = target_samples[idx]
        s = _extract_sample(dataset, real_idx)
        tid = s['tid']
        
        if not s['valid'] or tid not in seq_dict:
            continue
            
        cds_start, cds_end = s['cds_start_0'], s['cds_end_0']
        seq = seq_dict[tid].upper()
        L = len(seq)
        
        # 将张量送入正确的 device
        se = torch.from_numpy(s['se']).float().unsqueeze(0).to(device).requires_grad_(True)
        ce = torch.from_numpy(s['ce']).float().unsqueeze(0).to(device)
        ev = torch.from_numpy(s['ev']).float().unsqueeze(0).to(device) if s['ev'] is not None else None
        
        with torch.enable_grad():
            resolved_expr = raw._resolve_expr_vector(cell_type=s['ct'], expr_vector=ev, batch_size=1).to(device)
            
            # [关键修复 2]: 显式传入 cell_type，并且直接传入原始 s['species']
            # 这样符合你的 TranslationBaseModel.forward() 签名要求
            out = raw.forward(
                seq_batch=se, 
                count_batch=ce, 
                cell_type=s['ct'], 
                expr_vector=resolved_expr, 
                species=s['species'], 
                head_names=['count']
            )
            
            pred = out['count'].get('profile', out['count']) if isinstance(out['count'], dict) else out['count']
            profile_loss = (pred[0, :, 0] ** 2).sum()
            
        profile_loss.backward()
        grad = se.grad[0].detach().cpu().numpy()
        sal_track = np.abs(grad).sum(axis=-1)  # 单条转录本的碱基级 Saliency 曲线
        
        # 寻找上下文
        for x_pos in x_pos_hotspots:
            # 反向映射找真实物理位置
            abs_pos = _inverse_metagene(x_pos, cds_start, cds_end, FIXED_CDS_LEN)
            
            if 0 <= abs_pos < L:
                # 关键过滤：只有当这个特定的物理位置在这个细胞系里确实“亮起”了，才切它！
                if sal_track[abs_pos] > np.percentile(sal_track, 80): 
                    ctx_s = max(0, abs_pos - context_radius)
                    ctx_e = min(L, abs_pos + context_radius + 1)
                    ctx_seq = seq[ctx_s:ctx_e]
                    
                    # 补齐边界序列长度，保证聚类对齐
                    if len(ctx_seq) == (context_radius * 2 + 1) and 'N' not in ctx_seq:
                        valid_contexts.append(ctx_seq)
                        
    print(f"Harvested {len(valid_contexts)} highly salient context sequences.")
    return valid_contexts


def plot_sequence_logo(sequences, title="Cell-Type Specific Motif"):
    """使用 logomaker 绘制信息量 Logo 图"""
    if not sequences:
        print(f"No sequences found for {title}.")
        return
        
    # 计算每个位置的信息熵矩阵 (PWM)
    df = logomaker.alignment_to_matrix(sequences=sequences, to_type='information')
    
    fig, ax = plt.subplots(figsize=(8, 3))
    logo = logomaker.Logo(df, ax=ax, font_name='Arial Rounded MT Bold')
    logo.style_spines(visible=False)
    logo.style_spines(spines=['left', 'bottom'], visible=True)
    ax.set_ylabel('Information (bits)')
    ax.set_xlabel('Relative Position')
    plt.title(title)
    plt.show()
# ============================================================
# Notebook Cell 1: Transcript-Level Physical Sequence Slicing
# ============================================================
import logging
# [Fix]: 静音 matplotlib 缺失字体的警告
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

def _slice_and_append(seq, pos, r, target_list, tid, cds_start, cds_end):
    """辅助切片函数，确保边界完整且不含异常碱基，同时记录所有的位置元数据"""
    if pos - r >= 0 and pos + r + 1 <= len(seq):
        fragment = seq[pos - r : pos + r + 1]
        if 'N' not in fragment:
            rel_start = pos - cds_start
            rel_stop = pos - cds_end
            x_pos, _, _ = _map_to_metagene(pos, cds_start, cds_end, FIXED_CDS_LEN)
            
            # 将原来简单的字符串，改为字典记录结构，方便后续转成表
            target_list.append({
                'tid': tid,
                'sequence': fragment,
                'abs_pos': pos,
                'rel_to_cds_start': rel_start,
                'rel_to_cds_end': rel_stop,
                'x_pos': x_pos
            })


def extract_raw_peaks_by_region(model, dataset, seq_dict, out_dir, n_samples=300, window_radius=10, 
                                min_len=0, max_len=float('inf'), device=None):
    """
    Iterate through transcripts, compute absolute physical position saliency scores,
    split by 5' UTR, CDS, 3' UTR, and extract the strongest peak context.
    
    [Optimized]: 
    1. CDS region is expanded by `window_radius` upstream and downstream to naturally 
       capture boundary motifs (e.g., Kozak and Stop codon contexts) within the CDS group.
    2. Implements 1D Non-Maximum Suppression (NMS) to prevent overlapping extraction.
    
    Automatically saves individual raw dataframes to CSV files in the out_dir.
    """
    import os
    import numpy as np
    import pandas as pd
    import torch
    from tqdm import tqdm

    os.makedirs(out_dir, exist_ok=True)

    raw = _unwrap(model)
    if device is None:
        device = next(raw.parameters()).device
    raw.eval()

    region_sequences = {"5UTR": [], "CDS": [], "3UTR": []}
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

    for idx in tqdm(indices, desc="Extracting raw physical peaks"):
        s = _extract_sample(dataset, idx)
        tid = s['tid']
        if not s['valid'] or tid not in seq_dict:
            continue

        seq = seq_dict[tid].upper()
        L = len(seq)
        
        if L < min_len or L > max_len:
            continue
            
        cds_start = s['cds_start_0']
        cds_end = s['cds_end_0']

        se = torch.from_numpy(s['se']).float().unsqueeze(0).to(device).requires_grad_(True)
        ce = torch.from_numpy(s['ce']).float().unsqueeze(0).to(device)
        ev = torch.from_numpy(s['ev']).float().unsqueeze(0).to(device) if s['ev'] is not None else None

        with torch.enable_grad():
            resolved_expr = raw._resolve_expr_vector(cell_type=s['ct'], expr_vector=ev, batch_size=1).to(device)
            out = raw.forward(seq_batch=se, count_batch=ce, cell_type=s['ct'], expr_vector=resolved_expr, species=s['species'], head_names=['count'])
            pred = out['count'].get('profile', out['count']) if isinstance(out['count'], dict) else out['count']
            profile_loss = (pred[0, :, 0] ** 2).sum()

        profile_loss.backward()
        grad = se.grad[0].detach().cpu().numpy()
        sal_track = np.abs(grad).sum(axis=-1)

        # 使用 np.convolve 计算 3bp 窗口的滑动平均，减少单碱基梯度的抖动
        window_size = 3
        sal_track = np.convolve(sal_track, np.ones(window_size)/window_size, mode='same')

        # ==========================================
        # Step 1: Define Expanded Boundaries
        # ==========================================
        # 将 CDS 区域向外围扩展 window_radius 的长度
        # 从而把 Kozak 和 Stop context 划归到 CDS 搜索空间
        utr5_bound = max(0, cds_start)
        utr3_bound = min(L, cds_end)

        # ==========================================
        # Step 2: Collect Candidate Peaks per Region
        # ==========================================
        candidate_peaks = []

        # 5' UTR Region candidate (Strictly distal to expanded CDS)
        if utr5_bound > 0:
            utr5_sal = sal_track[:utr5_bound]
            if len(utr5_sal) > 0:
                peak_5 = np.argmax(utr5_sal)
                if utr5_sal[peak_5] > np.percentile(sal_track, 75): 
                    candidate_peaks.append((peak_5, utr5_sal[peak_5], "5UTR"))

        # CDS Region candidate (Includes the extended boundary contexts)
        if utr3_bound > utr5_bound:
            cds_sal = sal_track[utr5_bound:utr3_bound]
            if len(cds_sal) > 0:
                peak_cds = np.argmax(cds_sal) + utr5_bound
                if sal_track[peak_cds] > np.percentile(sal_track, 75):
                    candidate_peaks.append((peak_cds, sal_track[peak_cds], "CDS"))

        # 3' UTR Region candidate (Strictly distal to expanded CDS)
        if utr3_bound < L:
            utr3_sal = sal_track[utr3_bound:]
            if len(utr3_sal) > 0:
                peak_3 = np.argmax(utr3_sal) + utr3_bound
                if sal_track[peak_3] > np.percentile(sal_track, 75):
                    candidate_peaks.append((peak_3, sal_track[peak_3], "3UTR"))

        # ==========================================
        # Step 3: 1D Non-Maximum Suppression (NMS)
        # ==========================================
        # Sort candidates by absolute saliency score (Highest first)
        candidate_peaks.sort(key=lambda x: x[1], reverse=True)
        
        selected_peaks = []
        for pos, score, region in candidate_peaks:
            conflict = False
            for sel_pos, _, _ in selected_peaks:
                # If distance is <= 2 * window_radius, extraction windows overlap
                if abs(pos - sel_pos) <= 2 * window_radius:
                    conflict = True
                    break
            
            # Keeps the peak if it doesn't overlap with stronger peaks
            if not conflict:
                selected_peaks.append((pos, score, region))

        # ==========================================
        # Step 4: Slice and Append Validated Peaks
        # ==========================================
        for pos, _, region in selected_peaks:
            _slice_and_append(seq, pos, window_radius, region_sequences[region], tid, cds_start, cds_end)

    # Convert to DataFrames and automatically save to disk
    region_dfs = {}
    for region, data in region_sequences.items():
        if data:
            df = pd.DataFrame(data)
            df['Region'] = region
        else:
            df = pd.DataFrame(columns=['tid', 'sequence', 'abs_pos', 'rel_to_cds_start', 'rel_to_cds_end', 'x_pos', 'Region'])
        
        csv_filename = os.path.join(out_dir, f"raw_peaks_{region}.csv")
        df.to_csv(csv_filename, index=False)
        region_dfs[region] = df

    print(f"\nSuccessfully sliced & saved raw CSVs: 5'UTR ({len(region_dfs['5UTR'])}), CDS ({len(region_dfs['CDS'])}), 3'UTR ({len(region_dfs['3UTR'])})")
    return region_dfs


def extract_attn_peaks_by_region(model, dataset, seq_dict, out_dir, n_samples=300, window_radius=10, 
                                 min_len=0, max_len=float('inf'), device=None):
    """
    Iterate through transcripts, compute absolute physical position Attention scores,
    split by 5' UTR, CDS, 3' UTR, and extract the strongest peak context based on internal attention.
    
    [Optimized]: 
    1. Switched from Saliency (gradients) to Internal Multi-Head Attention weights.
    2. CDS region is expanded by `window_radius` upstream and downstream to naturally 
       capture boundary motifs (e.g., Kozak and Stop codon contexts) within the CDS group.
    3. Implements 1D Non-Maximum Suppression (NMS) to prevent overlapping extraction.
    
    Automatically saves individual raw dataframes to CSV files in the out_dir.
    """
    import os
    import numpy as np
    import pandas as pd
    import torch
    from tqdm import tqdm

    os.makedirs(out_dir, exist_ok=True)

    raw = _unwrap(model)
    if device is None:
        device = next(raw.parameters()).device
    raw.eval()

    region_sequences = {"5UTR": [], "CDS": [], "3UTR": []}
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

    for idx in tqdm(indices, desc="Extracting raw physical ATTN peaks"):
        s = _extract_sample(dataset, idx)
        tid = s['tid']
        if not s['valid'] or tid not in seq_dict:
            continue

        seq = seq_dict[tid].upper()
        L = len(seq)
        
        if L < min_len or L > max_len:
            continue
            
        cds_start = s['cds_start_0']
        cds_end = s['cds_end_0']

        # No requires_grad needed for Attention extraction
        se = torch.from_numpy(s['se']).float().unsqueeze(0).to(device)
        ce = torch.from_numpy(s['ce']).float().unsqueeze(0).to(device)
        ev = torch.from_numpy(s['ev']).float().unsqueeze(0).to(device) if s['ev'] is not None else None

        with torch.no_grad():
            # 1. Resolve environmental context exactly as in standard forward pass
            resolved_expr = raw._resolve_expr_vector(cell_type=s['ct'], expr_vector=ev, batch_size=1).to(device)
            species_idx = raw._normalize_species(s['species'], 1).to(device)
            species_emb = raw.species_embedding(species_idx)
            combined_env = torch.cat([resolved_expr, species_emb], dim=-1)
            compact_style = raw.expr_projector(combined_env)

            # 2. Extract input embeddings and masking
            src_reps = raw.src_emb(se, ce)
            src_mask = (se[:, :, 0] != 0).to(device)
            
            Lc = se.shape[1] # Length with padding
            n_heads = raw.n_heads
            head_dim = raw.encoder.encoder_layers[0].multi_headed_attention.head_dim
            
            attn_track = np.zeros(Lc)

            # 3. Manually step through encoder layers to intercept Attention Weights
            for enc_layer in raw.encoder.encoder_layers:
                sub = enc_layer.sublayers[0]
                style = sub.adaLN_modulation(compact_style)
                gamma, beta, alpha = style.chunk(3, dim=-1)
                normed = (1 + gamma.unsqueeze(1)) * sub.LN(src_reps) + beta.unsqueeze(1)

                attn_mod = enc_layer.multi_headed_attention
                bs_, _, d = normed.shape

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

                # Calculate "Received Attention": sum over query origins, average across heads
                received = attn_w.sum(dim=2).mean(dim=1)[0].cpu().numpy()
                attn_track += received

                # Continue the forward pass to feed the next layer
                attn_out = torch.matmul(attn_w, v)
                attn_out = attn_out.transpose(1, 2).reshape(bs_, Lc, n_heads * head_dim)
                attn_out = attn_mod.unifyheads(attn_out)
                if hasattr(attn_mod, 'dropout'):
                    attn_out = attn_mod.dropout(attn_out)
                src_reps = src_reps + alpha.unsqueeze(1) * sub.dropout(attn_out)

                sub2 = enc_layer.sublayers[1]
                style2 = sub2.adaLN_modulation(compact_style)
                gamma2, beta2, alpha2 = style2.chunk(3, dim=-1)
                normed2 = (1 + gamma2.unsqueeze(1)) * sub2.LN(src_reps) + beta2.unsqueeze(1)
                ffn_out = sub2.dropout(enc_layer.ffn(normed2))
                src_reps = src_reps + alpha2.unsqueeze(1) * ffn_out

        # Truncate to actual transcript length and calculate mean attention per layer
        attn_track = attn_track[:L] / len(raw.encoder.encoder_layers)

        # Smooth the track to avoid peak jittering
        # window_size = 3
        # attn_track = np.convolve(attn_track, np.ones(window_size)/window_size, mode='same')

        # ==========================================
        # Step 1: Define Expanded Boundaries
        # ==========================================
        # [Fixed]: Re-added the -/+ window_radius to correctly capture boundary motifs into CDS
        utr5_bound = max(0, cds_start - window_radius)
        utr3_bound = min(L, cds_end + window_radius)

        # ==========================================
        # Step 2: Collect Candidate Peaks per Region
        # ==========================================
        candidate_peaks = []

        if utr5_bound > 0:
            utr5_attn = attn_track[:utr5_bound]
            if len(utr5_attn) > 0:
                peak_5 = np.argmax(utr5_attn)
                if utr5_attn[peak_5] > np.percentile(attn_track, 75): 
                    candidate_peaks.append((peak_5, utr5_attn[peak_5], "5UTR"))

        if utr3_bound > utr5_bound:
            cds_attn = attn_track[utr5_bound:utr3_bound]
            if len(cds_attn) > 0:
                peak_cds = np.argmax(cds_attn) + utr5_bound
                if attn_track[peak_cds] > np.percentile(attn_track, 75):
                    candidate_peaks.append((peak_cds, attn_track[peak_cds], "CDS"))

        if utr3_bound < L:
            utr3_attn = attn_track[utr3_bound:]
            if len(utr3_attn) > 0:
                peak_3 = np.argmax(utr3_attn) + utr3_bound
                if attn_track[peak_3] > np.percentile(attn_track, 75):
                    candidate_peaks.append((peak_3, attn_track[peak_3], "3UTR"))

        # ==========================================
        # Step 3: 1D Non-Maximum Suppression (NMS)
        # ==========================================
        candidate_peaks.sort(key=lambda x: x[1], reverse=True)
        
        selected_peaks = []
        for pos, score, region in candidate_peaks:
            conflict = False
            for sel_pos, _, _ in selected_peaks:
                if abs(pos - sel_pos) <= 2 * window_radius:
                    conflict = True
                    break
            
            if not conflict:
                selected_peaks.append((pos, score, region))

        # ==========================================
        # Step 4: Slice and Append Validated Peaks
        # ==========================================
        for pos, _, region in selected_peaks:
            _slice_and_append(seq, pos, window_radius, region_sequences[region], tid, cds_start, cds_end)

    # Convert to DataFrames and automatically save to disk
    region_dfs = {}
    for region, data in region_sequences.items():
        if data:
            df = pd.DataFrame(data)
            df['Region'] = region
        else:
            df = pd.DataFrame(columns=['tid', 'sequence', 'abs_pos', 'rel_to_cds_start', 'rel_to_cds_end', 'x_pos', 'Region'])
        
        csv_filename = os.path.join(out_dir, f"raw_ATTN_peaks_{region}.csv")
        df.to_csv(csv_filename, index=False)
        region_dfs[region] = df

    print(f"\nSuccessfully sliced & saved raw ATTN CSVs: 5'UTR ({len(region_dfs['5UTR'])}), CDS ({len(region_dfs['CDS'])}), 3'UTR ({len(region_dfs['3UTR'])})")
    return region_dfs

def cluster_and_visualize_region_motifs(region_dfs, region_name, out_dir, min_clusters=4, max_clusters=10):
    """
    Perform unsupervised KMeans clustering on short physical sequences of a region.
    [Auto-K with Bounds]: Dynamically determines the optimal number of clusters 
    using Silhouette Score within a user-defined range [min_clusters, max_clusters].
    Appends cluster identifiers, saves to CSV, and exports PDF sequence logos.
    """
    import os
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import matplotlib.pyplot as plt
    import numpy as np
    import logomaker
    import pandas as pd
    
    os.makedirs(out_dir, exist_ok=True)

    df = region_dfs.get(region_name, pd.DataFrame())
    
    # Check if the dataframe contains enough sequences to support the minimum clustering request
    if len(df) < min_clusters * 3:
        print(f"Skipping cluster for {region_name}: Insufficient sequences ({len(df)}) to form {min_clusters} clusters.")
        return df.copy()

    sequences = df['sequence'].tolist()

    # Digitization of bases via One-Hot encoding
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encoded_list = []
    for seq in sequences:
        int_meta = [char_map[c] for c in seq]
        one_hot = np.eye(4)[int_meta].flatten() 
        encoded_list.append(one_hot)
    
    X = np.array(encoded_list)
    
    # =========================================================
    # Auto-K Determination with Strict Bounds Check
    # =========================================================
    # Determine the maximum safe upper limit based on available sample size
    safe_max_limit = min(max_clusters + 1, len(X) // 3 + 1)
    
    best_k = min_clusters
    best_score = -1.0
    best_labels = None
    
    # If the database size allows searching within the bound
    if safe_max_limit > min_clusters + 1:
        for k in range(min_clusters, safe_max_limit):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
            score = silhouette_score(X, kmeans.labels_)
            
            # Keep track of the highest Silhouette Score
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = kmeans.labels_
    else:
        # Fallback to min_clusters if the data size strictly restricts search space
        print(f"[{region_name}] Sample size is too tightly constrained. Forcing min_clusters={min_clusters}.")
        kmeans = KMeans(n_clusters=min_clusters, random_state=42, n_init=10).fit(X)
        best_k = min_clusters
        best_labels = kmeans.labels_

    print(f"\n========================================================")
    print(f"  [Auto-K Bound] {region_name} Optimal Clusters: {best_k} (Range: {min_clusters}-{max_clusters}, Score: {best_score:.3f})")
    print(f"========================================================")

    # Append results back to the dataframe copy
    df = df.copy()
    df['Cluster_ID'] = best_labels
    df['Motif_Name'] = region_name + "_Cluster" + df['Cluster_ID'].astype(str)

    # Save labeled dataset
    csv_save_path = os.path.join(out_dir, f"clustered_data_{region_name}.csv")
    df.to_csv(csv_save_path, index=False)

    # Visual rendering of sequence logos
    for cluster_id in range(best_k):
        sub_seqs = df[df['Cluster_ID'] == cluster_id]['sequence'].tolist()
        if len(sub_seqs) < 5: 
            continue
            
        counts_df = logomaker.alignment_to_matrix(sequences=sub_seqs, to_type='information')
        
        fig, ax = plt.subplots(figsize=(7, 2.2))
        logo = logomaker.Logo(counts_df, ax=ax)
        logo.style_spines(visible=False)
        logo.style_spines(spines=['left', 'bottom'], visible=True)
        
        ax.set_ylabel('Information (bits)')
        ax.set_xlabel('Relative Offset from Peak')
        ax.set_title(f"{region_name} - Cluster {cluster_id} (Support: n={len(sub_seqs)})")
        
        pdf_filename = os.path.join(out_dir, f"motif_logo_{region_name}_cluster_{cluster_id}.pdf")
        plt.savefig(pdf_filename, bbox_inches='tight', dpi=300)
        
        plt.show()
        plt.close(fig)
        
    return df


# ============================================================
# Notebook Cell 3: Motif Spatial Probability Heatmap
# ============================================================
def plot_motif_metagene_heatmap(
        all_motifs_df, out_path="motif_metagene_heatmap.pdf", 
        bin_size=20, up_len=300, down_len=300, max_prob=None,
        weight=8, height=10
        ):
    """
    Plots a heatmap of motif spatial distribution along the metagene.
    Y-axis motifs are dynamically sorted from 5' to 3' based on their peak enrichment positions.
    """
    import pandas as pd
    import numpy as np
    from plotnine import (ggplot, aes, geom_tile, geom_vline, scale_fill_gradient, 
                          labs, theme_classic, theme, element_text, element_blank, element_line)
    
    if 'Motif_Name' not in all_motifs_df.columns or all_motifs_df.empty:
        print("No motif clustering data available to plot.")
        return

    # 1. Filter coordinate boundaries
    df_plot = all_motifs_df[(all_motifs_df['x_pos'] >= -up_len) & (all_motifs_df['x_pos'] <= FIXED_CDS_LEN + down_len)].copy()
    
    # 2. Perform positional binning
    df_plot['x_bin'] = (df_plot['x_pos'] // bin_size) * bin_size + (bin_size / 2)
    
    # 3. Calculate raw frequencies
    heatmap_data = df_plot.groupby(['Motif_Name', 'x_bin']).size().reset_index(name='count')
    
    # 4. Impute empty grid tiles to guarantee a continuous matrix background
    unique_motifs = heatmap_data['Motif_Name'].unique()
    min_bin = (-up_len // bin_size) * bin_size + (bin_size / 2)
    max_bin = ((FIXED_CDS_LEN + down_len) // bin_size) * bin_size + (bin_size / 2)
    all_bins = np.arange(min_bin, max_bin + bin_size, bin_size)
    
    full_index = pd.MultiIndex.from_product([unique_motifs, all_bins], names=['Motif_Name', 'x_bin'])
    full_df = pd.DataFrame(index=full_index).reset_index()
    
    heatmap_data = pd.merge(full_df, heatmap_data, on=['Motif_Name', 'x_bin'], how='left').fillna({'count': 0})
    
    # 5. Standardize via row-wise probability
    motif_totals = heatmap_data.groupby('Motif_Name')['count'].transform('sum')
    heatmap_data['Probability'] = heatmap_data['count'] / (motif_totals + 1e-9)

    # 6. [核心重构]: 按照 5' -> 3' 空间富集峰值 (Peak) 对 Motif 进行动态排序
    # 找出每个 Motif 概率最大的 x_bin
    peak_bins = heatmap_data.loc[heatmap_data.groupby('Motif_Name')['Probability'].idxmax()]
    
    # 按 x_bin 降序排列 (因为 plotnine 的 Y 轴是从下往上画的，降序能让 5' 端最小的值排在图表最上面)
    # 如果有多个 Motif 在同一个位置 Peak，使用 Motif_Name 作为次级排序条件保持稳定
    ordered_motifs = peak_bins.sort_values(
        ['x_bin', 'Motif_Name'], 
        ascending=[False, False]
    )['Motif_Name'].tolist()

    # 应用严格排序
    heatmap_data['Motif_Name'] = pd.Categorical(
        heatmap_data['Motif_Name'], 
        categories=ordered_motifs
    )
    
    # 7. Automatically upscale the color scale if the global probability is too low
    if max_prob is None:
        max_prob = heatmap_data['Probability'].max()
        if max_prob == 0:
            max_prob = 1.0
    
    # 8. Render plot configurations
    p = (
        ggplot(heatmap_data, aes(x='x_bin', y='Motif_Name', fill='Probability'))
        + geom_tile(color='white', size=0.1) 
        + scale_fill_gradient(low='#EFF3FF', high='#08306B', limits=(0, max_prob)) 
        + geom_vline(xintercept=[0, FIXED_CDS_LEN], linetype='dashed', color='red', size=0.6)
        + labs(
            x=f'Metagene Position (Bin Size = {bin_size} nt)', 
            y='Discovered Motifs (Sorted 5\' \u2192 3\')', 
            fill='Spatial\nProbability', 
            title='Motif Spatial Distribution along Metagene'
        )
        + theme_classic()
        + theme(
            figure_size=(weight, height),
            axis_text_y=element_text(size=10),
            axis_text_x=element_text(size=10),
            axis_ticks_major_x=element_line(color='#333333', size=0.5),
            axis_ticks_major_y=element_line(color='#333333', size=0.5),
            axis_line_x=element_blank(),
            axis_line_y=element_blank()
        )
    )
    
    p.save(out_path)
    print(f"Motif Metagene Heatmap saved to {out_path}")\
    