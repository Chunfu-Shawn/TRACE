"""Jupyter notebook cells for matched RBP-motif translation analysis."""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 1: Imports and output directory                         ║
# ╚══════════════════════════════════════════════════════════════╝
"""
import os
import pickle
import pandas as pd

from eval.rbp_translation_effect import (
    run_rbp_translation_effect_analysis,
    run_targeted_rbp_saturation_mutagenesis,
)
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
# ║ Cell 5: Load canonical nucleotide-contribution results       ║
# ╚══════════════════════════════════════════════════════════════╝
"""
# These tables are reusable and do not require model inference.
rbp_contributions = pd.read_csv(
    os.path.join(out_dir, "rbp_nucleotide_contributions.csv")
)
rbp_effect_summary = pd.read_csv(
    os.path.join(out_dir, "rbp_motif_effect_summary.csv")
)

candidate_columns = [
    "Hit_ID",
    "Tid",
    "RBP_Name",
    "Region",
    "Motif_Start",
    "Motif_End",
    "Motif_Delta_Log2_TE",
    "Group_Median_Delta_Log2_TE",
    "Group_N_Transcripts",
]

available_case_hits = (
    rbp_contributions[candidate_columns]
    .drop_duplicates("Hit_ID")
    .assign(
        Absolute_Effect=lambda table: table["Motif_Delta_Log2_TE"].abs()
    )
    .sort_values("Absolute_Effect", ascending=False)
)

display(available_case_hits)
print(
    f"Available cases: {len(available_case_hits)} hits from "
    f"{available_case_hits['RBP_Name'].nunique()} RBPs"
)
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 5b: Plot strongest cases for selected RBPs              ║
# ╚══════════════════════════════════════════════════════════════╝
"""
requested_case_rbps = [
    "FXR1",
    "YBX3",
    "MCM3AP",
]

# Only RBPs already present in the contribution table can be plotted directly.
available_case_rbps = set(available_case_hits["RBP_Name"].astype(str))
selected_case_rbps = [
    rbp for rbp in requested_case_rbps if rbp in available_case_rbps
]
missing_case_rbps = [
    rbp for rbp in requested_case_rbps if rbp not in available_case_rbps
]
if missing_case_rbps:
    print("Contribution cases not yet calculated:", missing_case_rbps)
if not selected_case_rbps:
    raise ValueError(
        "None of requested_case_rbps are available in the contribution table."
    )

case_paths, selected_hits = plot_rbp_nucleotide_contribution_cases(
    contribution_df=rbp_contributions,
    out_dir=os.path.join(out_dir, "cases_selected_rbps"),
    summary_df=rbp_effect_summary,
    target_rbps=selected_case_rbps,
    target_regions=("5UTR", "3UTR"),
    cases_per_rbp=2,
    require_summary_direction=True,
    max_cases=None,
    return_selected_hits=True,
    width=8.0,
    height=3.4,
)

display(selected_hits[candidate_columns])
case_paths
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 5c: Plot one or more exact Hit_ID values                ║
# ╚══════════════════════════════════════════════════════════════╝
"""
# Copy one or more IDs from selected_hits or the candidate table above.
exact_hit_ids = [
    selected_hits.iloc[0]["Hit_ID"],
]

exact_case_paths, exact_hits = plot_rbp_nucleotide_contribution_cases(
    contribution_df=rbp_contributions,
    out_dir=os.path.join(out_dir, "cases_exact_hits"),
    target_hit_ids=exact_hit_ids,
    return_selected_hits=True,
    width=8.0,
    height=3.4,
)

display(exact_hits[candidate_columns])
exact_case_paths
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 5d: Select a case by RBP, region, transcript, position  ║
# ╚══════════════════════════════════════════════════════════════╝
"""
filtered_case_paths, filtered_hits = plot_rbp_nucleotide_contribution_cases(
    contribution_df=rbp_contributions,
    out_dir=os.path.join(out_dir, "cases_position_filtered"),
    summary_df=rbp_effect_summary,
    target_rbps=[selected_case_rbps[0]],
    target_regions=None,  # For example: ["5UTR"]
    target_transcript_ids=None,  # For example: ["ENST00000381348"]
    target_motif_starts=None,  # For example: [125]
    cases_per_rbp=3,
    require_summary_direction=True,
    return_selected_hits=True,
)

display(filtered_hits[candidate_columns])
filtered_case_paths
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 5e: Select any hit from the complete hit-effect table   ║
# ╚══════════════════════════════════════════════════════════════╝
"""
# The hit-effect table contains all evaluated motif hits, including hits that
# were not selected for the original representative-case analysis.
rbp_hit_effects = pd.read_csv(
    os.path.join(out_dir, "rbp_motif_hit_effects.csv")
)

with open(
    os.path.join(out_dir, "unique_transcript_samples.pkl"),
    "rb",
) as handle:
    transcript_samples = pickle.load(handle)

with open(
    os.path.join(out_dir, "validated_rbp_pwms.pkl"),
    "rb",
) as handle:
    validated_rbp_pwms = pickle.load(handle)

target_rbp = "HNRNPA1"
target_region = "5UTR"
target_tid = None  # For example: "ENST00000436324"
target_motif_start = None  # For example: 51

candidate_effects = rbp_hit_effects[
    (rbp_hit_effects["RBP_Name"].astype(str) == target_rbp)
    & (rbp_hit_effects["Region"].astype(str) == target_region)
].copy()
if target_tid is not None:
    candidate_effects = candidate_effects[
        candidate_effects["Tid"].astype(str) == str(target_tid)
    ]
if target_motif_start is not None:
    candidate_effects = candidate_effects[
        candidate_effects["Start"].astype(int) == int(target_motif_start)
    ]

candidate_effects = candidate_effects.assign(
    Absolute_Effect=candidate_effects["Delta_Log2_TE"].abs()
).sort_values("Absolute_Effect", ascending=False)

effect_columns = [
    "Hit_ID",
    "Tid",
    "RBP_Name",
    "Region",
    "Start",
    "End",
    "Motif_Sequence",
    "PWM_Score",
    "Delta_Log2_TE",
    "WT_CDS_Mean_Signal",
]
display(candidate_effects[effect_columns].head(20))
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 5f: Run standalone saturation mutagenesis               ║
# ╚══════════════════════════════════════════════════════════════╝
"""
if candidate_effects.empty:
    raise ValueError("No hit matches the requested RBP/transcript/position.")

# Select exact rows after reviewing the candidate table.
selected_effect_hits = candidate_effects.head(1)
selected_effect_hit_ids = selected_effect_hits["Hit_ID"].astype(str).tolist()
selected_hit = selected_effect_hits.iloc[0]

targeted_saturation_csv = os.path.join(
    out_dir,
    (
        f"targeted_saturation.{selected_hit['RBP_Name']}."
        f"{selected_hit['Tid']}.{selected_hit['Hit_ID']}.csv"
    ),
)

targeted_contributions = run_targeted_rbp_saturation_mutagenesis(
    model=model,
    hit_effects=rbp_hit_effects,
    samples=transcript_samples,
    pwm_library=validated_rbp_pwms,
    output_csv=targeted_saturation_csv,
    target_hit_ids=selected_effect_hit_ids,
    context_flank=20,
    prediction_scale="log1p",
    batch_size=64,
)

display(targeted_contributions.head(20))
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ Cell 5g: Plot signed native-base contribution letters        ║
# ╚══════════════════════════════════════════════════════════════╝
"""
targeted_case_paths, targeted_hits = (
    plot_rbp_nucleotide_contribution_cases(
        contribution_df=targeted_contributions,
        out_dir=os.path.join(out_dir, "cases_targeted_saturation"),
        target_hit_ids=selected_effect_hit_ids,
        return_selected_hits=True,
        width=8.0,
        height=3.4,
    )
)

display(targeted_hits[candidate_columns])
targeted_case_paths
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
