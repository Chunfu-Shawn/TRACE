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
    prediction_scale="log1p",
    force_zero_expression=True,
    batch_size=32,
    min_transcripts=5,
    n_cases_per_direction=3,
    case_selection_mode="global",
    case_regions=("5UTR", "3UTR"),
    de_novo_source="signed_attribution",
    de_novo_num_transcripts=500,
    de_novo_peaks_per_direction=1,
    de_novo_regions=("5UTR", "3UTR"),
    position_bin_size=20,
    position_utr5_length=300,
    position_cds_length=600,
    position_utr3_length=300,
    position_known_rbp_scope="all",
    random_state=42,
)

# Copy the canonical result files in out_dir between servers.
print(results["result_directory"])
display(results["summary"].head(20))
display(results["de_novo_motifs"].head(20))
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 3b: Reuse copied results on a plotting-only server       ║
# ╚══════════════════════════════════════════════════════════════╝
"""
# This requires all canonical raw-result files to already exist in out_dir.
# Model, dataset, and PWM inputs are not accessed when every stage is complete.
results = run_rbp_translation_effect_analysis(
    model=None,
    dataset=None,
    pwm_library=None,
    metadata=None,
    out_dir=out_dir,
)
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
# ║ Cell 4b: Plot selected RBP translation effects               ║
# ╚══════════════════════════════════════════════════════════════╝
"""
# Load the canonical summary directly; no model inference is required.
rbp_effect_summary = pd.read_csv(
    os.path.join(out_dir, "rbp_motif_effect_summary.csv")
)

selected_rbps = [
    "HNRNPA1",
    "ELAVL1",
    "PUM1",
    "PUM2",
]

plot_rbp_translation_effect_summary(
    rbp_effect_summary,
    out_path=os.path.join(
        out_dir,
        "rbp_translation_effect_summary.selected.pdf",
    ),
    target_rbps=selected_rbps,
    target_regions=("5UTR", "CDS", "3UTR"),
    fdr_threshold=None,
    width=6.2,
    row_height=0.30,
)
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 4c: Plot significant effects for selected RBPs           ║
# ╚══════════════════════════════════════════════════════════════╝
"""
plot_rbp_translation_effect_summary(
    rbp_effect_summary,
    out_path=os.path.join(
        out_dir,
        "rbp_translation_effect_significant.selected.pdf",
    ),
    target_rbps=selected_rbps,
    target_regions=("5UTR", "3UTR"),
    fdr_threshold=0.10,
    width=6.2,
    row_height=0.30,
)
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 5: Plot signed nucleotide-contribution cases            ║
# ╚══════════════════════════════════════════════════════════════╝
"""
case_paths = plot_rbp_nucleotide_contribution_cases(
    results["nucleotide_contributions"],
    out_dir=os.path.join(out_dir, "cases"),
    summary_df=results["summary"],
    cases_per_rbp=3,
    max_cases=None,
)
case_paths
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 5b: Plot exact RBP, region, transcript, or hit cases    ║
# ╚══════════════════════════════════════════════════════════════╝
"""
case_paths, selected_hits = plot_rbp_nucleotide_contribution_cases(
    results["nucleotide_contributions"],
    out_dir=os.path.join(out_dir, "cases_selected"),
    summary_df=results["summary"],
    target_rbps=["HNRNPA1"],
    target_regions=["5UTR"],
    target_hit_ids=None,  # For example: ["RBP_HIT_0000123"]
    target_transcript_ids=None,
    target_motif_starts=None,
    cases_per_rbp=3,
    return_selected_hits=True,
)
display(selected_hits)
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
        layout=(
            "combined" if profile_key == "known_rbp_position_profiles"
            else "regional_pages"
        ),
        vector_cells=(profile_key == "known_rbp_position_profiles"),
    )
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 7b: Plot selected known RBPs from canonical results      ║
# ╚══════════════════════════════════════════════════════════════╝
"""
# Load the reusable raw result directly; no model or motif scan is required.
known_position_path = os.path.join(
    out_dir,
    "known_rbp_position_profiles.csv",
)
known_position_profiles = pd.read_csv(known_position_path)

selected_rbps = [
    "HNRNPA1",
    "ELAVL1",
    "PUM1",
    "PUM2",
]

selected_heatmap_path = plot_motif_position_preference_heatmap(
    known_position_profiles,
    out_path=os.path.join(
        out_dir,
        "known_rbp_position_preference_heatmap.pdf",
    ),
    target_features=selected_rbps,
    cluster_mode="regions",  # Use "full" or "none" as alternatives.
    min_total_hits=1,
    max_features=0,
    value_col="Log2_Positional_Enrichment",
    width=7.2,
    row_height=0.22,
    layout="combined",
    vector_cells=True,
)
selected_heatmap_path
"""
