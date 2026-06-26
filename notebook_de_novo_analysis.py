"""
Jupyter notebook cells for de novo motif discovery.

Two-phase strategy:
  Phase 1 (fast, model-intrinsic): attention + saliency + gene attribution
  Phase 2 (targeted): only mutate hotspot positions identified in Phase 1

All position-based metrics are aligned to CDS start (pos 0 = first nt
of start codon), making them comparable across variable-length sequences.
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 1: Imports and setup                                    ║
# ╚══════════════════════════════════════════════════════════════╝
"""
import sys
sys.path.insert(0, "/path/to/TRACE/src")

import os, pickle
import numpy as np
import pandas as pd
import torch
from plotnine import *
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from eval.de_novo_motif_discovery import (
    _extract_sample,
    extract_attention_positional_importance,
    compute_saliency_profile,
    compute_adaLN_gene_attribution,
    identify_hotspot_positions,
    targeted_mutagenesis,
    aggregate_mutagenesis,
    extract_hotspot_contexts,
)

from eval.calculate_te import calculate_morf_mean_signal

out_dir = "./results/de_novo"
os.makedirs(out_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 2: Load model, data, metadata                          ║
# ╚══════════════════════════════════════════════════════════════╝
"""
# --- Model ---
# from model.translation_base_model import TranslationBaseModel
# model = TranslationBaseModel.from_pretrained("path/to/checkpoint.pt")
# model = model.to(device).eval()

# --- Dataset ---
# from data.translation_dataset import TranslationDataset
# dataset = TranslationDataset.from_h5("path/to/dataset.h5", lazy=False)

# --- Sequences ---
# with open("seq_dict.pkl", "rb") as f:
#     seq_dict = pickle.load(f)

# --- CDS metadata ---
# with open("tx_cds.pkl", "rb") as f:
#     tx_cds = pickle.load(f)

# --- Gene names (optional, for attribution) ---
# gene_names = [...]   # list of 16k symbols, or None

print(f"Model: {next(model.parameters()).device}, "
      f"layers={len(model.encoder.encoder_layers)}, "
      f"heads={model.n_heads}, d_expr={model.d_expr}")
print(f"Dataset: {len(dataset)} samples")
print(f"Sequences: {len(seq_dict)}, CDS: {len(tx_cds)}")
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 3: Phase 1A — Attention positional importance           ║
# ╚══════════════════════════════════════════════════════════════╝
"""
# Single forward pass per sample, CDS-start-aligned aggregation.
# Cost: ~5-10 min for 200 samples on A2.

attn_df = extract_attention_positional_importance(
    model, dataset, n_samples=200, max_len=1200, device=device,
)
attn_df.to_csv(os.path.join(out_dir, "attention_positional_importance.csv"), index=False)

# ---- Plot: per-layer attention received by position ----
p1 = (
    ggplot(attn_df, aes(x='pos_from_cds_start', y='mean_attn'))
    + geom_line(color='#2171B5', size=0.4, alpha=0.8)
    + facet_wrap('~layer', ncol=4, scales='free_y')
    + geom_vline(xintercept=0, linetype='dashed', color='#C44E52', size=0.4)
    + labs(x='Position relative to CDS start', y='Mean attention received')
    + theme_bw() + theme(figure_size=(18, 12))
)
p1.save(os.path.join(out_dir, "attention_by_layer.pdf"))

# ---- Plot: mean across all layers ----
combined = attn_df.groupby('pos_from_cds_start').agg(
    mean_attn=('mean_attn', 'mean'),
    sem=('mean_attn', lambda x: np.std(x) / np.sqrt(len(x)))
).reset_index()

p2 = (
    ggplot(combined, aes(x='pos_from_cds_start', y='mean_attn'))
    + geom_line(color='#08519C', size=1)
    + geom_ribbon(aes(ymin='mean_attn - sem', ymax='mean_attn + sem'),
                   alpha=0.15, fill='#08519C')
    + geom_vline(xintercept=0, linetype='dashed', color='#C44E52', size=0.8)
    + labs(x='Position relative to CDS start',
           y='Mean attention received (all layers)',
           title='Global attention positional importance')
    + theme_bw() + theme(figure_size=(14, 4))
)
p2.save(os.path.join(out_dir, "attention_combined.pdf"))

# Top attended positions
top_attn = combined.nlargest(10, 'mean_attn')
print("Top 10 attended positions (rel. to CDS start):")
print(top_attn[['pos_from_cds_start', 'mean_attn']])
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 4: Phase 1B — Input nucleotide saliency                 ║
# ╚══════════════════════════════════════════════════════════════╝
"""
# Single forward+backward pass per sample. Cost similar to Cell 3.

sal_df = compute_saliency_profile(
    model, dataset, n_samples=100, max_len=1200, device=device,
)
sal_df.to_csv(os.path.join(out_dir, "input_saliency_profile.csv"), index=False)

# Plot
p = (
    ggplot(sal_df, aes(x='pos_from_cds_start', y='mean_saliency'))
    + geom_line(color='#238B45', size=0.8)
    + geom_ribbon(aes(ymin='mean_saliency - std_saliency/np.sqrt(n)',
                       ymax='mean_saliency + std_saliency/np.sqrt(n)'),
                   alpha=0.15, fill='#238B45')
    + geom_vline(xintercept=0, linetype='dashed', color='#C44E52', size=0.8)
    + labs(x='Position relative to CDS start',
           y='Mean |d(TE)/d(base)|',
           title='Input nucleotide saliency for TE prediction')
    + theme_bw() + theme(figure_size=(14, 4))
)
p.save(os.path.join(out_dir, "saliency_profile.pdf"))

top_sal = sal_df.nlargest(15, 'mean_saliency')
print("Top 15 positions by saliency (rel. to CDS start):")
print(top_sal[['pos_from_cds_start', 'mean_saliency']])
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 5: Phase 1C — AdaLN gene attribution                    ║
# ╚══════════════════════════════════════════════════════════════╝
"""
# Pure weight inspection. Zero inference cost. Instant.

adaLN_df = compute_adaLN_gene_attribution(model, gene_names=gene_names, top_k=30)
adaLN_df.to_csv(os.path.join(out_dir, "adaLN_gene_attribution.csv"), index=False)

# Top genes summed across all layers
top_genes = adaLN_df.groupby('gene')['score'].sum().nlargest(25).index
plot_df = adaLN_df[adaLN_df['gene'].isin(top_genes)]

p = (
    ggplot(plot_df, aes(x='layer_module', y='gene', fill='score_norm'))
    + geom_tile()
    + scale_fill_gradient(low='white', high='#08519C', name='Norm.')
    + labs(x='Layer-Module', y='Gene',
           title='AdaLN gene attribution per layer-module')
    + theme_bw()
    + theme(axis_text_x=element_text(rotation=45, hjust=1),
            figure_size=(14, 8))
)
p.save(os.path.join(out_dir, "adaLN_gene_attribution_heatmap.pdf"))

# Print top genes
top_summary = adaLN_df.groupby('gene')['score'].sum().sort_values(ascending=False).head(25)
print("Top 25 genes by AdaLN attribution:")
for i, (g, s) in enumerate(top_summary.items()):
    print(f"  {i+1:2d}. {g}: {s:.4f}")
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 6: Identify hotspot positions from Phase 1              ║
# ╚══════════════════════════════════════════════════════════════╝
"""
# Combine attention and saliency to select positions for mutagenesis.

hotspot_positions, hotspot_df = identify_hotspot_positions(
    attn_df, sal_df, n_hotspots=30,
    layer_range=(max(0, len(model.encoder.encoder_layers)-4),
                 len(model.encoder.encoder_layers)-1),  # top layers
)
hotspot_df.to_csv(os.path.join(out_dir, "hotspot_positions.csv"), index=False)

# Plot combined score
p = (
    ggplot(hotspot_df, aes(x='pos_from_cds_start', y='combined_score'))
    + geom_col(fill='#8C2D04', alpha=0.7)
    + geom_vline(xintercept=0, linetype='dashed', color='#C44E52', size=0.8)
    + labs(x='Position relative to CDS start',
           y='Combined importance (attn + saliency Z-score)',
           title=f'Top {len(hotspot_positions)} hotspot positions')
    + theme_bw() + theme(figure_size=(14, 3))
)
p.save(os.path.join(out_dir, "hotspot_positions_bar.pdf"))

print(f"Selected {len(hotspot_positions)} hotspot positions for targeted mutagenesis:")
print(hotspot_positions)

# Kozak positions for reference
kozak = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 4, 5]
overlap = set(hotspot_positions) & set(kozak)
print(f"Overlap with Kozak positions: {sorted(overlap)}")
print(f"Novel (non-Kozak) hotspots: {sorted(set(hotspot_positions) - set(kozak))}")
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 7: Phase 2 — Targeted mutagenesis at hotspots           ║
# ╚══════════════════════════════════════════════════════════════╝
"""
# Only mutate the K hotspot positions, not the full sequence.
# Cost: n_transcripts × K × 3 forward passes.
# With n=30, K=30: ~2700 forward passes. Manageable on 2×A2.

mut_df = targeted_mutagenesis(
    model, dataset, seq_dict, tx_cds,
    target_positions=hotspot_positions,
    n_transcripts=30,
    device=device,
)
mut_df.to_csv(os.path.join(out_dir, "targeted_mutagenesis.csv"), index=False)

pos_agg, base_agg = aggregate_mutagenesis(mut_df)

# ---- Plot: mean |delta_TE| by hotspot position ----
p1 = (
    ggplot(pos_agg, aes(x='pos_from_cds_start', y='mean_abs_delta'))
    + geom_col(fill='#8C2D04', alpha=0.7)
    + geom_errorbar(aes(ymin='mean_abs_delta - sem',
                         ymax='mean_abs_delta + sem'),
                     width=0.5)
    + geom_vline(xintercept=0, linetype='dashed', color='#C44E52', size=0.8)
    + labs(x='Position relative to CDS start',
           y='Mean |Delta TE| (mutation impact)',
           title='Targeted mutagenesis: positional impact at hotspots')
    + theme_bw() + theme(figure_size=(14, 3.5))
)
p1.save(os.path.join(out_dir, "targeted_mutagenesis_impact.pdf"))

# ---- Plot: base-specific effects heatmap ----
base_agg['substitution'] = base_agg['ref_base'] + '→' + base_agg['mut_base']
base_agg['pos_from_cds_start'] = pd.Categorical(
    base_agg['pos_from_cds_start'],
    categories=sorted(base_agg['pos_from_cds_start'].unique())
)

p2 = (
    ggplot(base_agg, aes(x='pos_from_cds_start', y='substitution', fill='mean_delta'))
    + geom_tile()
    + scale_fill_gradient2(low='#2166AC', mid='white', high='#B2182B',
                           name='Mean ΔTE')
    + labs(x='Position relative to CDS start', y='Substitution',
           title='Base-specific mutational effects at hotspots')
    + theme_bw()
    + theme(axis_text_x=element_text(rotation=45, hjust=1),
            figure_size=(14, 6))
)
p2.save(os.path.join(out_dir, "targeted_mutagenesis_base_effects.pdf"))

print("Targeted mutagenesis complete")
print(pos_agg[['pos_from_cds_start', 'mean_abs_delta', 'n']])
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 8: Extract sequence contexts for MEME motif discovery   ║
# ╚══════════════════════════════════════════════════════════════╝
"""
# For hotspot positions with high mutagenesis impact, extract
# surrounding sequence contexts → feed to MEME for de novo motif discovery.

# Select top-impact hotspots
top_impact = pos_agg.nlargest(10, 'mean_abs_delta')['pos_from_cds_start'].tolist()
print(f"Extracting contexts for top {len(top_impact)} impact positions: {top_impact}")

contexts = extract_hotspot_contexts(
    seq_dict, tx_cds,
    hotspot_positions=top_impact,
    context_radius=20,
    max_seqs=200,
)

for rel_pos, ctx_list in contexts.items():
    if len(ctx_list) >= 5:
        fasta_path = os.path.join(out_dir, f"hotspot_context_pos{rel_pos}.fasta")
        with open(fasta_path, 'w') as f:
            for i, ctx in enumerate(ctx_list[:100]):
                f.write(f">ctx_{i}_pos{rel_pos}\n{ctx}\n")
        print(f"  pos {rel_pos:4d}: {len(ctx_list[:100])} sequences → {fasta_path}")

# Print consensus around the #1 hotspot
if contexts:
    from collections import Counter
    top_pos = top_impact[0]
    if top_pos in contexts and len(contexts[top_pos]) > 10:
        ctx_list = contexts[top_pos][:50]
        max_len = max(len(c) for c in ctx_list)
        consensus = ''
        for i in range(max_len):
            bases = [c[i] for c in ctx_list if i < len(c) and c[i] != 'N']
            consensus += Counter(bases).most_common(1)[0][0] if bases else '.'
        marker_pos = 20  # context_radius
        print(f"\nConsensus around pos {top_pos} (marker at position {marker_pos}):")
        print(f"  {consensus}")
        print(f"  {' ' * marker_pos}^")
        print(f"\n  → Upload hotspot_context_pos{top_pos}.fasta to https://meme-suite.org/meme/tools/meme")
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 9: Cross-validate with prior knowledge results          ║
# ╚══════════════════════════════════════════════════════════════╝
"""
# Overlay mutagenesis results with Kozak position labels.
# If model has internalized Kozak rules, -3 and +4 should show high impact.

kozak_positions = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 4, 5]
kozak_labels = pd.DataFrame({
    'pos_from_cds_start': kozak_positions,
    'label': ['-6', '-5', '-4', '-3', '-2', '-1', 'A', 'T', 'G', '+4', '+5']
})

# Merge with hotspot_df (has combined_score)
plot_df = hotspot_df.copy()

# Add column for Kozak status
plot_df['is_kozak'] = plot_df['pos_from_cds_start'].isin(kozak_positions)

p = (
    ggplot(plot_df, aes(x='pos_from_cds_start', y='combined_score',
                         fill='is_kozak'))
    + geom_col(alpha=0.8)
    + geom_text(data=kozak_labels[kozak_labels['pos_from_cds_start'].isin(plot_df['pos_from_cds_start'])],
                mapping=aes(x='pos_from_cds_start',
                           y=plot_df['combined_score'].max() * 1.08,
                           label='label'),
                size=8, color='#2171B5', inherit_aes=False)
    + scale_fill_manual(values={True: '#C44E52', False: '#8C2D04'},
                        labels={True: 'Kozak pos', False: 'Novel'})
    + geom_vline(xintercept=0, linetype='dashed', color='black', size=0.6)
    + labs(x='Position relative to CDS start',
           y='Combined importance (Z-score)',
           title='Hotspot importance with Kozak position overlay',
           fill='')
    + theme_bw() + theme(figure_size=(14, 4))
)
p.save(os.path.join(out_dir, "hotspots_with_kozak_overlay.pdf"))
print("Cross-validation plot saved")

# Summary statistics
novel = set(hotspot_positions) - set(kozak_positions)
kozak_hit = set(hotspot_positions) & set(kozak_positions)
print(f"Kozak positions recovered: {sorted(kozak_hit)} ({len(kozak_hit)}/{len(kozak_positions)})")
print(f"Novel hotspot positions:    {sorted(novel)} ({len(novel)})")
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 10: Pipeline summary                                    ║
# ╚══════════════════════════════════════════════════════════════╝
"""
print("=" * 60)
print("De novo motif discovery pipeline complete!")
print("=" * 60)
print(f"Results in: {out_dir}/")
print()
print("Phase 1 (model-intrinsic, fast):")
print("  attention_by_layer.pdf         — per-layer attention peaks")
print("  attention_combined.pdf         — all-layer mean attention")
print("  saliency_profile.pdf           — input gradient importance")
print("  adaLN_gene_attribution_heatmap.pdf — genes driving modulation")
print()
print("Phase 2 (targeted mutagenesis):")
print("  hotspot_positions_bar.pdf      — selected hotspot positions")
print("  targeted_mutagenesis_impact.pdf — mutation impact per hotspot")
print("  targeted_mutagenesis_base_effects.pdf — per-substitution effects")
print("  hotspots_with_kozak_overlay.pdf — cross-validation with Kozak")
print()
print("For MEME motif discovery:")
print("  hotspot_context_pos*.fasta     — upload to meme-suite.org")
print()
print("Next steps:")
print("  1. Does the model recover Kozak positions (-3, +4) as hotspots?")
print("     → Yes: model internalized known rules; novel hotspots are credible")
print("  2. Upload the top hotspot .fasta files to MEME for de novo motifs")
print("  3. Match MEME-discovered motifs to RBP databases (TOMTOM)")
print("  4. GO/KEGG enrichment on top AdaLN-attributed genes")
print("  5. For cell-type-specific analysis, run interpretability_analysis.py")
"""
