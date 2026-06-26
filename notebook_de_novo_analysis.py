"""
Jupyter notebook cells for de novo motif discovery.

Two-panel plots: CDS start (left) + CDS stop (right), per-layer LOESS curves.
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
    compute_saliency_profile,
    compute_adaLN_gene_attribution,
    identify_hotspot_positions,
    targeted_mutagenesis,
    aggregate_mutagenesis,
    extract_hotspot_contexts,
    plot_attention_profile,
    plot_saliency_profile,
    plot_mutagenesis_profile,
)

out_dir = "./results/de_novo"; os.makedirs(out_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 2: Load model and data                                  ║
# ╚══════════════════════════════════════════════════════════════╝
"""
# model = ...          # trained model
# dataset = ...        # TranslationDataset
# seq_dict = ...       # {tid: "ACGT..."}
# tx_cds = ...         # {tid: {cds_start_pos, cds_end_pos, ...}}
# gene_names = [...]   # optional

raw = _unwrap(model)
print(f"Layers={len(raw.encoder.encoder_layers)}, heads={raw.n_heads}, "
      f"d_expr={raw.d_expr}, d_species={raw.d_species}")
print(f"Dataset: {len(dataset)} samples")
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 3: Phase 1A — Attention positional importance           ║
# ╚══════════════════════════════════════════════════════════════╝
"""
attn_df_start, attn_df_stop, avg_cds_len = extract_attention_positional_importance(
    model, dataset, n_samples=200, max_len=1200, device=device,
)
attn_df_start.to_csv(os.path.join(out_dir, "attention_start_aligned.csv"), index=False)
attn_df_stop.to_csv(os.path.join(out_dir, "attention_stop_aligned.csv"), index=False)

# Two-panel: CDS start (left) + CDS stop (right), 12 layers colored
# Uses real cds_end_pos from dataset metadata for stop alignment
plot_attention_profile(
    attn_df_start, attn_df_stop,
    out_path=os.path.join(out_dir, "attention_profile_start_stop.pdf"),
    start_region=(-100, 300),
    stop_region=(-300, 100),
)

# Top positions near CDS start
near = attn_df_start[attn_df_start['pos_from_cds_start'].between(-15, 15)]
top = near.groupby('pos_from_cds_start')['mean_attn'].mean().nlargest(10)
print("Top attention near CDS start:", dict(top))
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 4: Phase 1B — Input saliency                            ║
# ╚══════════════════════════════════════════════════════════════╝
"""
sal_df = compute_saliency_profile(
    model, dataset, n_samples=100, max_len=1200, device=device,
)
sal_df.to_csv(os.path.join(out_dir, "input_saliency_profile.csv"), index=False)

plot_saliency_profile(
    sal_df,
    avg_cds_len=avg_cds_len,  # from Cell 3
    out_path=os.path.join(out_dir, "saliency_profile_start_stop.pdf"),
)

top_sal = sal_df.dropna(subset=['pos_from_cds_start']).nlargest(15, 'mean_saliency')
print("Top saliency:", dict(zip(top_sal['pos_from_cds_start'], top_sal['mean_saliency'].round(6))))
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 5: Phase 1C — AdaLN gene attribution                    ║
# ╚══════════════════════════════════════════════════════════════╝
"""
adaLN_df = compute_adaLN_gene_attribution(model, gene_names=gene_names, top_k=30)
adaLN_df.to_csv(os.path.join(out_dir, "adaLN_gene_attribution.csv"), index=False)

from plotnine import *
top_genes = adaLN_df.groupby('gene')['score'].sum().nlargest(25).index
(
    ggplot(adaLN_df[adaLN_df['gene'].isin(top_genes)],
           aes(x='layer_module', y='gene', fill='score_norm'))
    + geom_tile()
    + scale_fill_gradient(low='white', high='#08519C')
    + labs(x='Layer-Module', y='Gene', title='AdaLN gene attribution')
    + theme_bw()
    + theme(axis_text_x=element_text(rotation=45, hjust=1), figure_size=(14, 8))
).save(os.path.join(out_dir, "adaLN_gene_attribution_heatmap.pdf"))

top = adaLN_df.groupby('gene')['score'].sum().sort_values(ascending=False).head(25)
print("Top 25 genes:"); [print(f"  {i+1:2d}. {g}: {s:.4f}") for i,(g,s) in enumerate(top.items())]
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 6: Identify hotspots                                    ║
# ╚══════════════════════════════════════════════════════════════╝
"""
n_layers = len(raw.encoder.encoder_layers)
hotspot_positions, hotspot_df = identify_hotspot_positions(
    attn_df, sal_df, n_hotspots=30,
    layer_range=(max(0, n_layers - 4), n_layers - 1),
)
hotspot_df.to_csv(os.path.join(out_dir, "hotspot_positions.csv"), index=False)

kozak = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 4, 5]
hotspot_df['is_kozak'] = hotspot_df['pos_from_cds_start'].astype(int).isin(kozak)
(
    ggplot(hotspot_df, aes(x='pos_from_cds_start', y='combined_score', fill='is_kozak'))
    + geom_col(alpha=0.8)
    + scale_fill_manual(values={True: '#C44E52', False: '#8C2D04'})
    + geom_vline(xintercept=0, linetype='dashed')
    + labs(x='Position relative to CDS start', y='Combined Z-score', fill='')
    + theme_bw() + theme(figure_size=(14, 4))
).save(os.path.join(out_dir, "hotspots_with_kozak.pdf"))

novel = set(hotspot_positions) - set(kozak)
print(f"Kozak recovered: {sorted(set(hotspot_positions) & set(kozak))}")
print(f"Novel hotspots:  {sorted(novel)}")
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 7: Phase 2 — Targeted mutagenesis                       ║
# ╚══════════════════════════════════════════════════════════════╝
"""
mut_df = targeted_mutagenesis(
    model, dataset, seq_dict, tx_cds,
    target_positions=hotspot_positions,
    n_transcripts=30, device=device,
)
mut_df.to_csv(os.path.join(out_dir, "targeted_mutagenesis.csv"), index=False)

pos_agg, base_agg = aggregate_mutagenesis(mut_df)

plot_mutagenesis_profile(
    pos_agg,
    avg_cds_len=avg_cds_len,
    out_path=os.path.join(out_dir, "mutagenesis_impact_start_stop.pdf"),
)

# Base-specific heatmap
base_agg['sub'] = base_agg['ref_base'] + '\u2192' + base_agg['mut_base']
(
    ggplot(base_agg, aes(x='pos_from_cds_start', y='sub', fill='mean_delta'))
    + geom_tile()
    + scale_fill_gradient2(low='#2166AC', mid='white', high='#B2182B')
    + geom_vline(xintercept=0, linetype='dashed')
    + labs(x='Position relative to CDS start', y='Substitution')
    + theme_bw() + theme(figure_size=(14, 6))
).save(os.path.join(out_dir, "mutagenesis_base_effects.pdf"))

print(pos_agg[['pos_from_cds_start', 'mean_abs_delta', 'n']].head(15))
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 8: MEME context extraction                              ║
# ╚══════════════════════════════════════════════════════════════╝
"""
top_impact = pos_agg.nlargest(10, 'mean_abs_delta')['pos_from_cds_start'].tolist()
contexts = extract_hotspot_contexts(seq_dict, tx_cds, top_impact, context_radius=20, max_seqs=200)

for rel_pos, ctx_list in contexts.items():
    if len(ctx_list) >= 5:
        fasta = os.path.join(out_dir, f"hotspot_context_pos{rel_pos}.fasta")
        with open(fasta, 'w') as f:
            for i, ctx in enumerate(ctx_list[:100]):
                f.write(f">ctx_{i}_pos{rel_pos}\n{ctx}\n")
        print(f"  pos {rel_pos:4d}: {len(ctx_list[:100])} seqs -> {fasta}")

print("\n-> Upload .fasta to https://meme-suite.org/meme/tools/meme")
"""
