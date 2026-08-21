"""
Jupyter notebook cells for RBP regulatory landscape analysis.

Pipeline:
  1. Split transcripts by TE into top-20% / bottom-20%, extract attention peaks.
  2. Scan peaks against unified RBP PWM library.
  3. Plot bubble chart: normalized attention vs TE enrichment.
  4. Plot metagene spatial heatmap for top RBPs.
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 1: Imports and setup                                    ║
# ╚══════════════════════════════════════════════════════════════╝
"""
import sys; sys.path.insert(0, "/path/to/TRACE/src")
import os, pickle
import numpy as np, pandas as pd
import torch
from tqdm import tqdm
import warnings; warnings.filterwarnings("ignore")

from eval.de_novo_motif_discovery import (
    _unwrap,
    extract_attention_positional_importance,
    split_and_extract_contrastive_peaks,
    extract_attn_peaks_by_region,
)
from eval.rbp_scan import (
    parse_cisbp_pwms,
    load_cisbp_metadata,
    parse_attract_pwms,
    rbp_centric_peak_scanner,
    score_and_map_peaks,
)
from plot.de_novo_motif_discovery import plot_attention_profile
from plot.rbp_scan import (
    plot_rbp_metagene_heatmap,
    plot_rbp_regulatory_bubble,
)

out_dir = "./results/rbp_scan"; os.makedirs(out_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 2: Load model, dataset, seq_dict, tx_cds                ║
# ╚══════════════════════════════════════════════════════════════╝
"""
# model = ...          # trained / fine-tuned model
# dataset = ...        # TranslationDataset
# seq_dict = ...       # {tid: "ACGT..."}  full transcript sequences
# tx_cds = ...         # {tid: {cds_start_pos, cds_end_pos, ...}}

raw = _unwrap(model)
print(f"Layers={len(raw.encoder.encoder_layers)}, heads={raw.n_heads}, "
      f"d_expr={raw.d_expr}")
print(f"Dataset: {len(dataset)} samples")
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 3: Phase 1A — Attention positional importance           ║
# ╚══════════════════════════════════════════════════════════════╝
"""
attn_df = extract_attention_positional_importance(
    model, dataset, n_samples=500, min_len=500, max_len=1200, device=device,
)
attn_df.to_csv(os.path.join(out_dir, "attention_positional_importance.csv"), index=False)

plot_attention_profile(
    attn_df,
    out_path=os.path.join(out_dir, "attention_profile.pdf"),
    up_len=300, down_len=300,
)

# Quick check: mean attention by metagene region
attn_df['region'] = pd.cut(
    attn_df['x_pos'],
    bins=[-float('inf'), 0, 600, float('inf')],
    labels=['5UTR', 'CDS', '3UTR']
)
print(attn_df.groupby('region')['mean_attn'].mean())
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 4: Split by TE, extract attention peaks per group       ║
# ╚══════════════════════════════════════════════════════════════╝
"""
top_ratio = 0.20
min_len, max_len = 500, 4000
attn_perc = 75      # percentile threshold for peak inclusion
window_radius = 10

high_te_dfs, low_te_dfs, transcript_te_dict = split_and_extract_contrastive_peaks(
    model, dataset, seq_dict,
    out_dir=out_dir,
    min_len=min_len, max_len=max_len,
    attn_perc=attn_perc,
    top_ratio=top_ratio,
    window_radius=window_radius,
    device=device,
)

# Print summary
for region in ['5UTR', 'CDS', '3UTR']:
    n_high = len(high_te_dfs.get(region, pd.DataFrame()))
    n_low = len(low_te_dfs.get(region, pd.DataFrame()))
    print(f"  {region}: High-TE peaks={n_high}, Low-TE peaks={n_low}")
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 5: Load RBP PWM libraries (CISBP + ATtRACT)             ║
# ╚══════════════════════════════════════════════════════════════╝
"""
# --- CISBP-RNA ---
cisbp_pwm_dir = "/path/to/CISBP/RNA/PWM/directory"
cisbp_info_path = "/path/to/CISBP/RBP_Information_all_motifs.txt"
cisbp_pwms = parse_cisbp_pwms(cisbp_pwm_dir)
cisbp_meta = load_cisbp_metadata(cisbp_info_path)

# --- ATtRACT ---
attract_pwm_path = "/path/to/ATtRACT/pwm.txt"
attract_pwms = parse_attract_pwms(attract_pwm_path)
# ATtRACT metadata: assumes a pre-built CSV with columns Matrix_id, Gene_name, ...
attract_meta = pd.read_csv("/path/to/ATtRACT/metadata.csv")
attract_meta = attract_meta.rename(columns={
    'Matrix_id': 'Matrix_id', 'Gene_name': 'Gene_name'
})

# --- Merge ---
# Keep only CISBP Gene_name/Matrix_id colums to align with ATtRACT
cisbp_meta_sub = cisbp_meta[['Matrix_id', 'Gene_name']].copy()
cisbp_meta_sub['Database'] = 'CISBP'
attract_meta_sub = attract_meta[['Matrix_id', 'Gene_name']].copy()
attract_meta_sub['Database'] = 'ATtRACT'

unified_meta = pd.concat([cisbp_meta_sub, attract_meta_sub], ignore_index=True)
unified_pwms = {**cisbp_pwms, **attract_pwms}

print(f"Unified PWM library: {len(unified_pwms)} matrices, "
      f"{unified_meta['Gene_name'].nunique()} unique RBPs")
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 6: RBP-centric scan — High-TE vs Low-TE peaks           ║
# ╚══════════════════════════════════════════════════════════════╝
"""
# Scan RBPs against BOTH High- and Low-TE attention peaks.
# Enrichment_Ratio = (High_Hits + 1) / (Low_Hits + 1).
# Only RBPs from High-TE peaks contribute to Mean_Attention.
rbp_landscape_df = rbp_centric_peak_scanner(
    high_te_dfs, low_te_dfs,
    unified_pwms, unified_meta,
    out_dir=out_dir,
    min_match_score=0.85,
)

print(f"\nRBPs with >= 5 total hits: {len(rbp_landscape_df)}")
print(rbp_landscape_df[['RBP_Name', 'High_Hits', 'Low_Hits',
                         'Mean_Attention', 'Enrichment_Ratio']].head(15))
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 7: Regulatory bubble plot                               ║
# ╚══════════════════════════════════════════════════════════════╝
"""
plot_rbp_regulatory_bubble(
    rbp_landscape_df,
    out_path=os.path.join(out_dir, "rbp_regulatory_bubble.pdf"),
    top_n_label=15,
)

# Print top RBPs in upper-right quadrant
upper_right = rbp_landscape_df[
    (rbp_landscape_df['Mean_Attention'] > rbp_landscape_df['Mean_Attention'].median()) &
    (rbp_landscape_df['Enrichment_Ratio'] > 1.0)
].nlargest(20, 'Total_Hits')

print("\nTop RBPs (upper-right quadrant):")
for _, r in upper_right.iterrows():
    print(f"  {r['RBP_Name']:20s}  hits={r['Total_Hits']:5d}  "
          f"attn={r['Mean_Attention']:.3f}  enrich={r['Enrichment_Ratio']:.2f}")
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 8: Metagene spatial heatmap for top RBPs (optional)     ║
# ╚══════════════════════════════════════════════════════════════╝
"""
# Map each peak to its matching RBPs for spatial resolution
master_peaks = pd.concat(
    [df for df in high_te_dfs.values() if not df.empty],
    ignore_index=True,
)
mapped_peaks_df = score_and_map_peaks(
    master_peaks, unified_pwms, unified_meta, min_match_score=0.85,
)

# Spatial heatmap for top-30 RBPs by hits
top_rbps = rbp_landscape_df.nlargest(30, 'Total_Hits')['RBP_Name'].tolist()
mapped_subset = mapped_peaks_df[mapped_peaks_df['RBP_Name'].isin(top_rbps)]

plot_rbp_metagene_heatmap(
    mapped_subset,
    out_path=os.path.join(out_dir, "rbp_metagene_heatmap_top30.pdf"),
    bin_size=20, up_len=300, down_len=300,
)
"""
