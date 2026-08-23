"""Jupyter notebook cells for matched RBP-motif translation analysis."""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 1: Imports and output directory                         ║
# ╚══════════════════════════════════════════════════════════════╝
"""
import os
import pickle
import pandas as pd

from eval.rbp_translation_effect import run_rbp_translation_effect_analysis
from plot.rbp_scan import (
    plot_motif_position_preference_heatmap,
    plot_rbp_translation_effect_summary,
    plot_rbp_nucleotide_contribution_cases,
    plot_de_novo_translation_motif_logos,
)

out_dir = "/path/to/results/rbp_translation_effect"
os.makedirs(out_dir, exist_ok=True)
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 2: Load the known RBP PWM library                       ║
# ╚══════════════════════════════════════════════════════════════╝
"""
with open("/path/to/Unified_RBP_PWMs.pkl", "rb") as handle:
    rbp_pwms = pickle.load(handle)

rbp_metadata = pd.read_csv(
    "/path/to/Unified_RBP_Metadata_Annotated.tsv",
    sep="\t",
)

# Optional: start with a biologically focused subset before a full scan.
target_rbps = None
# target_rbps = ["EIF4A3", "ELAVL1", "PUM1", "PUM2", "IGF2BP1"]
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 3: Run matched motif disruption and discovery           ║
# ╚══════════════════════════════════════════════════════════════╝
"""
results = run_rbp_translation_effect_analysis(
    model=model,
    dataset=test_dataset,
    pwm_library=rbp_pwms,
    metadata=rbp_metadata,
    out_dir=out_dir,
    target_rbps=target_rbps,
    target_transcript_ids=None,
    regions=("5UTR", "CDS", "3UTR"),
    num_transcripts=2000,
    score_threshold=0.85,
    max_hits_per_rbp_transcript_region=1,
    context_flank=12,
    known_motif_scan_workers=8,
    scan_backend="process",
    scan_chunk_size=None,
    reuse_known_motif_scan=True,
    known_motif_scan_cache_path=os.path.join(
        out_dir, "known_rbp_motif_hits.pkl"
    ),
    prediction_scale="log1p",
    force_zero_expression=True,
    batch_size=32,
    min_transcripts=5,
    n_cases_per_direction=3,
    de_novo_source="signed_attribution",
    de_novo_num_transcripts=500,
    de_novo_peaks_per_direction=1,
    position_bin_size=20,
    position_utr5_length=300,
    position_cds_length=600,
    position_utr3_length=300,
    position_known_rbp_scope="all",
    random_state=42,
)

# Move both the PKL and its .manifest.json sidecar between servers.
print(results["known_motif_scan_cache_path"])
display(results["summary"].head(20))
display(results["de_novo_motifs"].head(20))
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 4: Plot positive and negative RBP motif effects         ║
# ╚══════════════════════════════════════════════════════════════╝
"""
plot_rbp_translation_effect_summary(
    results["summary"],
    out_path=os.path.join(out_dir, "rbp_translation_effect_summary.pdf"),
    top_n_per_direction=30,
    fdr_threshold=0.10,
)
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 5: Plot signed nucleotide-contribution cases            ║
# ╚══════════════════════════════════════════════════════════════╝
"""
case_paths = plot_rbp_nucleotide_contribution_cases(
    results["nucleotide_contributions"],
    out_dir=os.path.join(out_dir, "cases"),
    max_cases=6,
)
case_paths
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 6: Plot de novo motif logos                             ║
# ╚══════════════════════════════════════════════════════════════╝
"""
plot_de_novo_translation_motif_logos(
    results["de_novo_motifs"],
    results["de_novo_alignments"],
    out_path=os.path.join(out_dir, "de_novo_translation_motif_logos.pdf"),
    top_n_per_direction=4,
)
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 7: Plot known-RBP and de novo positional preferences    ║
# ╚══════════════════════════════════════════════════════════════╝
"""
for profile_key, filename in [
    (
        "known_rbp_position_profiles",
        "known_rbp_position_preference_heatmap.pdf",
    ),
    (
        "de_novo_position_profiles",
        "de_novo_position_preference_heatmap.pdf",
    ),
]:
    profile_df = results[profile_key]
    if profile_df.empty:
        continue
    plot_motif_position_preference_heatmap(
        profile_df,
        out_path=os.path.join(out_dir, filename),
        cluster_mode="regions",  # Use "full" or "none" as alternatives.
        min_total_hits=1,
        max_features=0,
        value_col="Log2_Positional_Enrichment",
        width=7.2,
        row_height=0.07,
    )
"""
