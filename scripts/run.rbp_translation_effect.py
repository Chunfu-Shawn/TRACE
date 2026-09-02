#!/usr/bin/env python3
"""Run the file-resumable TRACE RBP translation-effect motif pipeline."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.translation_dataset import TranslationDataset
from eval.rbp_translation_effect import (
    RBPMotifMutagenesisEvaluator,
    build_motif_position_profiles,
    collect_rbp_motif_hits,
    collect_unique_transcript_samples,
    discover_de_novo_translation_motifs,
    extract_signed_translation_attribution_windows,
    summarize_rbp_motif_effects,
    validate_rbp_pwm_library,
)
from model.base_model import BaseModel
from model.prediction_heads import PsiteDensityHead
STAGES = (
    "validate_pwms",
    "samples",
    "hits",
    "effects",
    "summary",
    "cases",
    "attribution",
    "de_novo",
    "positions",
    "plots",
)


def _json_default(value: Any):
    """Convert common scientific Python values to JSON-safe objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON.")


def _atomic_pickle(value: Any, path: Path) -> None:
    """Atomically serialize a public binary result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _atomic_json(value: Any, path: Path) -> None:
    """Atomically serialize a public JSON result or manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, default=_json_default)
    os.replace(temporary, path)


def _atomic_csv(table: pd.DataFrame, path: Path) -> None:
    """Atomically save a public CSV result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _load_result_csv(path: Path, stage: str) -> pd.DataFrame:
    """Load a canonical CSV result, including a valid zero-row result."""
    try:
        table = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        table = pd.DataFrame()
    print(f"[SKIP] {stage}: loaded {path.name}")
    return table


def _load_result_pickle(path: Path, stage: str):
    """Load a canonical pickle result."""
    with path.open("rb") as handle:
        value = pickle.load(handle)
    print(f"[SKIP] {stage}: loaded {path.name}")
    return value


def _load_result_json(path: Path, stage: str):
    """Load a canonical JSON result."""
    value = json.loads(path.read_text(encoding="utf-8"))
    print(f"[SKIP] {stage}: loaded {path.name}")
    return value


def _load_dataset(paths: list[str]):
    """Load one or more pickle/HDF5 translation datasets lazily when possible."""
    datasets = []
    for path_text in paths:
        path = Path(path_text).expanduser().resolve()
        suffix = path.suffix.lower()
        if suffix in {".h5", ".hdf5"}:
            dataset = TranslationDataset.from_h5(str(path), lazy=True)
        elif suffix in {".pkl", ".pickle"}:
            dataset = TranslationDataset.from_pickle(str(path))
        else:
            raise ValueError(
                f"Unsupported dataset format '{suffix}' for {path}; "
                "use .h5, .hdf5, .pkl, or .pickle."
            )
        print(f"Loaded dataset: {path} ({len(dataset):,} rows)")
        datasets.append(dataset)
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def _load_pickle(path: str):
    with Path(path).expanduser().open("rb") as handle:
        return pickle.load(handle)


def _load_id_collection(path: Optional[str]) -> Optional[list[str]]:
    """Load IDs from pickle, JSON, CSV/TSV, or one-ID-per-line text."""
    if path is None:
        return None
    source = Path(path).expanduser().resolve()
    suffix = source.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        value = _load_pickle(str(source))
    elif suffix in {".pt", ".pth"}:
        value = torch.load(source, map_location="cpu")
    elif suffix == ".json":
        value = json.loads(source.read_text(encoding="utf-8"))
    elif suffix in {".csv", ".tsv"}:
        table = pd.read_csv(source, sep="\t" if suffix == ".tsv" else ",")
        preferred = next(
            (
                column for column in
                ("Tid", "tid", "Transcript_ID", "transcript_id", "Gene_name")
                if column in table.columns
            ),
            table.columns[0],
        )
        value = table[preferred].dropna().astype(str).tolist()
    else:
        value = [
            line.strip()
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if isinstance(value, Mapping):
        flattened = []
        for items in value.values():
            if isinstance(items, str):
                flattened.append(items)
            else:
                try:
                    flattened.extend(items)
                except TypeError:
                    flattened.append(items)
        value = flattened
    if isinstance(value, str):
        value = [value]
    return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


def _parse_csv_values(values: Optional[list[str]]) -> Optional[list[str]]:
    """Split repeatable comma-separated command-line values."""
    if not values:
        return None
    parsed = []
    for value in values:
        parsed.extend(part.strip() for part in value.split(",") if part.strip())
    return list(dict.fromkeys(parsed))


def _load_model(args, device: torch.device) -> BaseModel:
    """Construct BaseModel, attach its count head, and restore a checkpoint."""
    model = BaseModel.from_config(str(Path(args.model_config).expanduser()))
    model.add_head(
        "count",
        PsiteDensityHead.create_from_model(
            model,
            d_pred_h=args.head_hidden_dim,
        ),
        overwrite=True,
        move_to_model_device=False,
    )
    model.to(device)
    checkpoint = torch.load(
        str(Path(args.checkpoint).expanduser()),
        map_location=device,
    )
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and all(
        isinstance(value, torch.Tensor) for value in checkpoint.values()
    ):
        state_dict = checkpoint
    else:
        raise ValueError(f"Unsupported checkpoint format: {args.checkpoint}")
    state_dict = model._strip_head_module_prefix(state_dict)
    load_result = model.load_state_dict(state_dict, strict=not args.non_strict)
    if args.non_strict:
        print(
            "Non-strict checkpoint load: "
            f"missing={load_result.missing_keys}, "
            f"unexpected={load_result.unexpected_keys}"
        )
    model.eval()
    print(f"Loaded model on {device}: {args.checkpoint}")
    return model


def _device_from_argument(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def _should_stop(stage: str, stop_after: Optional[str]) -> bool:
    if stop_after == stage:
        print(f"Stopped after requested stage: {stage}")
        return True
    return False


def build_parser() -> argparse.ArgumentParser:
    """Build the cluster-friendly command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "File-resumable known-RBP perturbation and de novo translation "
            "motif analysis for TRACE BaseModel checkpoints."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    required = parser.add_argument_group("input paths")
    required.add_argument(
        "--model-config",
        help="Required only when a missing stage needs model inference.",
    )
    required.add_argument(
        "--checkpoint",
        help="Required only when a missing stage needs model inference.",
    )
    required.add_argument(
        "--dataset",
        action="append",
        help="Required only when unique_transcript_samples.pkl is missing.",
    )
    required.add_argument(
        "--pwm-pkl",
        help="Required only when validated_rbp_pwms.pkl is missing.",
    )
    required.add_argument(
        "--metadata-tsv",
        help="Required only when validation or motif scanning must run.",
    )
    required.add_argument("--out-dir", required=True)

    model_group = parser.add_argument_group("model and inference")
    model_group.add_argument("--device", default="auto")
    model_group.add_argument("--head-hidden-dim", type=int, default=384)
    model_group.add_argument("--non-strict", action="store_true")
    model_group.add_argument("--prediction-scale", choices=["log1p", "linear"], default="log1p")
    model_group.add_argument("--batch-size", type=int, default=32)
    model_group.add_argument(
        "--use-dataset-expression",
        action="store_true",
        help="Use dataset expression vectors instead of zero expression conditioning.",
    )

    selection = parser.add_argument_group("sample and motif selection")
    selection.add_argument("--target-rbp", action="append")
    selection.add_argument("--target-rbp-file")
    selection.add_argument("--target-transcript-file")
    selection.add_argument("--regions", nargs="+", default=["5UTR", "CDS", "3UTR"])
    selection.add_argument("--num-transcripts", type=int, default=2000)
    selection.add_argument("--score-threshold", type=float, default=0.85)
    selection.add_argument("--max-hits-per-rbp-transcript-region", type=int, default=1)
    selection.add_argument("--context-flank", type=int, default=12)
    selection.add_argument(
        "--scan-workers",
        type=int,
        default=1,
        help=(
            "Worker threads or processes used to scan independent transcripts "
            "for RBP motifs."
        ),
    )
    selection.add_argument(
        "--scan-backend",
        choices=["thread", "process"],
        default="thread",
        help="Parallel execution backend for known-RBP motif scanning.",
    )
    selection.add_argument(
        "--scan-chunk-size",
        type=int,
        help=(
            "Transcripts submitted per process task; by default this is "
            "calculated to create approximately four chunks per worker."
        ),
    )
    selection.add_argument("--random-state", type=int, default=42)

    statistics = parser.add_argument_group("statistics and discovery")
    statistics.add_argument("--min-transcripts", type=int, default=5)
    statistics.add_argument("--bootstrap-iterations", type=int, default=2000)
    statistics.add_argument("--confidence-level", type=float, default=0.95)
    statistics.add_argument("--n-cases-per-direction", type=int, default=3)
    statistics.add_argument(
        "--case-selection-mode",
        choices=["global", "per_rbp"],
        default="global",
        help=(
            "Use the legacy global representative selection or calculate "
            "the strongest cases independently for every eligible RBP."
        ),
    )
    statistics.add_argument(
        "--case-regions",
        nargs="+",
        choices=["5UTR", "CDS", "3UTR"],
        default=["5UTR", "3UTR"],
        help="Regions eligible for representative nucleotide-contribution cases.",
    )
    statistics.add_argument(
        "--de-novo-source",
        choices=["signed_attribution", "known_hit_context"],
        default="signed_attribution",
    )
    statistics.add_argument("--de-novo-num-transcripts", type=int, default=500)
    statistics.add_argument("--de-novo-peaks-per-direction", type=int, default=1)
    statistics.add_argument("--de-novo-k", nargs="+", type=int, default=[5, 6, 7, 8])
    statistics.add_argument("--de-novo-extreme-quantile", type=float, default=0.75)
    statistics.add_argument("--de-novo-neutral-quantile", type=float, default=0.40)
    statistics.add_argument("--de-novo-min-occurrences", type=int, default=5)
    statistics.add_argument("--de-novo-top-n-per-direction", type=int, default=10)
    statistics.add_argument("--de-novo-logo-flank", type=int, default=10)
    statistics.add_argument(
        "--de-novo-regions",
        nargs="+",
        choices=["5UTR", "CDS", "3UTR"],
        default=["5UTR", "3UTR"],
        help="Regions used for attribution peaks and region-matched discovery.",
    )

    plotting = parser.add_argument_group("PDF plotting")
    plotting.add_argument("--skip-plots", action="store_true")
    plotting.add_argument("--plot-top-n-per-direction", type=int, default=30)
    plotting.add_argument("--plot-fdr-threshold", type=float)
    plotting.add_argument("--plot-max-cases", type=int)
    plotting.add_argument("--plot-cases-per-rbp", type=int, default=3)
    plotting.add_argument("--plot-case-rbp", action="append")
    plotting.add_argument(
        "--plot-case-region",
        action="append",
        choices=["5UTR", "CDS", "3UTR"],
    )
    plotting.add_argument("--plot-case-hit-id", action="append")
    plotting.add_argument("--plot-case-transcript-id", action="append")
    plotting.add_argument("--plot-case-motif-start", action="append", type=int)
    plotting.add_argument("--plot-logo-top-n", type=int, default=4)
    plotting.add_argument(
        "--position-cluster-mode",
        choices=["regions", "full", "none"],
        default="regions",
    )
    plotting.add_argument("--position-bin-size", type=int, default=20)
    plotting.add_argument("--position-utr5-length", type=int, default=300)
    plotting.add_argument("--position-cds-length", type=int, default=600)
    plotting.add_argument("--position-utr3-length", type=int, default=300)
    plotting.add_argument(
        "--position-bins-per-region",
        type=int,
        default=None,
        help="Legacy equal-region bin count; overrides the three fixed lengths.",
    )
    plotting.add_argument("--position-min-hits", type=int, default=1)
    plotting.add_argument(
        "--position-max-features",
        type=int,
        default=0,
        help="Maximum heatmap rows; use 0 to show every retained feature.",
    )
    plotting.add_argument(
        "--position-rbp-scope",
        choices=["summary", "all"],
        default="all",
        help="Use statistically summarized RBPs or every scanned RBP.",
    )
    plotting.add_argument("--position-pseudocount", type=float, default=0.5)
    plotting.add_argument("--position-heatmap-width", type=float, default=7.2)
    plotting.add_argument("--position-row-height", type=float, default=0.07)

    execution = parser.add_argument_group("execution control")
    execution.add_argument("--stop-after", choices=STAGES)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Execute stages, reusing canonical result files when they exist."""
    args = build_parser().parse_args(argv)
    if args.de_novo_logo_flank < 1:
        raise ValueError("--de-novo-logo-flank must be positive.")
    if args.de_novo_logo_flank > args.context_flank:
        raise ValueError(
            "--de-novo-logo-flank cannot exceed --context-flank for "
            "peak-centered logos."
        )
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _device_from_argument(args.device)

    _atomic_json(
        {
            "arguments": vars(args),
            "stages": list(STAGES),
            "reuse_policy": (
                "Canonical output files are authoritative. Delete a stage's "
                "output files to rerun that stage."
            ),
        },
        out_dir / "rbp_pipeline_run_manifest.json",
    )

    model_holder = {}
    dataset_holder = {}
    input_holder = {}
    selection_holder = {}

    def require_input(value, option: str, stage: str):
        if value is None or value == []:
            raise ValueError(
                f"{option} is required because the '{stage}' result is missing."
            )
        return value

    def get_pwm_library():
        if "pwm_library" not in input_holder:
            pwm_path = require_input(
                args.pwm_pkl, "--pwm-pkl", "validate_pwms"
            )
            value = _load_pickle(pwm_path)
            if not isinstance(value, dict):
                raise TypeError(
                    "--pwm-pkl must contain a dictionary keyed by Matrix_id."
                )
            input_holder["pwm_library"] = value
        return input_holder["pwm_library"]

    def get_metadata():
        if "metadata" not in input_holder:
            metadata_path = require_input(
                args.metadata_tsv, "--metadata-tsv", "hits"
            )
            input_holder["metadata"] = pd.read_csv(
                metadata_path,
                sep="\t",
                dtype={"Matrix_id": str},
            )
        return input_holder["metadata"]

    def get_target_rbps():
        if "target_rbps" not in selection_holder:
            values = _parse_csv_values(args.target_rbp)
            from_file = _load_id_collection(args.target_rbp_file)
            if from_file:
                values = list(dict.fromkeys((values or []) + from_file))
            selection_holder["target_rbps"] = values
        return selection_holder["target_rbps"]

    def get_target_transcripts():
        if "target_transcripts" not in selection_holder:
            selection_holder["target_transcripts"] = _load_id_collection(
                args.target_transcript_file
            )
        return selection_holder["target_transcripts"]

    def get_model():
        if "model" not in model_holder:
            require_input(args.model_config, "--model-config", "inference")
            require_input(args.checkpoint, "--checkpoint", "inference")
            model_holder["model"] = _load_model(args, device)
        return model_holder["model"]

    def get_dataset():
        if "dataset" not in dataset_holder:
            dataset_paths = require_input(args.dataset, "--dataset", "samples")
            dataset_holder["dataset"] = _load_dataset(dataset_paths)
        return dataset_holder["dataset"]

    pwm_audit_path = out_dir / "rbp_pwm_validation.csv"
    valid_pwms_path = out_dir / "validated_rbp_pwms.pkl"
    if pwm_audit_path.is_file() and valid_pwms_path.is_file():
        pwm_audit = _load_result_csv(pwm_audit_path, "validate_pwms")
        valid_pwms = _load_result_pickle(valid_pwms_path, "validate_pwms")
    else:
        print("[RUN] validate_pwms")
        valid_pwms, pwm_audit = validate_rbp_pwm_library(
            get_pwm_library(),
            metadata=get_metadata(),
            target_rbps=get_target_rbps(),
        )
        _atomic_csv(pwm_audit, pwm_audit_path)
        _atomic_pickle(valid_pwms, valid_pwms_path)
        print("[DONE] validate_pwms: canonical results saved")
    if _should_stop("validate_pwms", args.stop_after):
        return 0

    samples_path = out_dir / "unique_transcript_samples.pkl"
    if samples_path.is_file():
        samples = _load_result_pickle(samples_path, "samples")
    else:
        print("[RUN] samples")
        samples = collect_unique_transcript_samples(
            get_dataset(),
            target_transcript_ids=get_target_transcripts(),
            num_transcripts=args.num_transcripts,
            random_state=args.random_state,
        )
        if not args.use_dataset_expression:
            for sample in samples.values():
                sample["Expr_Vector"] = np.zeros_like(sample["Expr_Vector"])
        _atomic_pickle(samples, samples_path)
        print("[DONE] samples: unique_transcript_samples.pkl saved")
    if _should_stop("samples", args.stop_after):
        return 0

    hits_path = out_dir / "rbp_motif_hits.csv"
    if hits_path.is_file():
        hits = _load_result_csv(hits_path, "hits")
    else:
        print("[RUN] hits")
        hits = collect_rbp_motif_hits(
            samples,
            valid_pwms,
            get_metadata(),
            target_rbps=get_target_rbps(),
            regions=args.regions,
            score_threshold=args.score_threshold,
            max_hits_per_rbp_transcript_region=(
                args.max_hits_per_rbp_transcript_region
            ),
            context_flank=args.context_flank,
            num_workers=args.scan_workers,
            scan_backend=args.scan_backend,
            scan_chunk_size=args.scan_chunk_size,
        )
        _atomic_csv(hits, hits_path)
        print("[DONE] hits: rbp_motif_hits.csv saved")
    if _should_stop("hits", args.stop_after):
        return 0

    effects_path = out_dir / "rbp_motif_hit_effects.csv"
    if effects_path.is_file():
        effects = _load_result_csv(effects_path, "effects")
    else:
        print("[RUN] effects")
        if hits.empty:
            effects = pd.DataFrame()
        else:
            evaluator = RBPMotifMutagenesisEvaluator(
                get_model(),
                valid_pwms,
                prediction_scale=args.prediction_scale,
            )
            effects = evaluator.evaluate_hits(
                hits,
                samples,
                batch_size=args.batch_size,
            )
            effects["Expression_Conditioning"] = (
                "dataset" if args.use_dataset_expression else "zero"
            )
        _atomic_csv(effects, effects_path)
        print("[DONE] effects: rbp_motif_hit_effects.csv saved")
    if _should_stop("effects", args.stop_after):
        return 0

    summary_path = out_dir / "rbp_motif_effect_summary.csv"
    if summary_path.is_file():
        summary = _load_result_csv(summary_path, "summary")
    else:
        print("[RUN] summary")
        summary = (
            pd.DataFrame()
            if effects.empty
            else summarize_rbp_motif_effects(
                effects,
                min_transcripts=args.min_transcripts,
                bootstrap_iterations=args.bootstrap_iterations,
                confidence_level=args.confidence_level,
                random_state=args.random_state,
            )
        )
        _atomic_csv(summary, summary_path)
        print("[DONE] summary: rbp_motif_effect_summary.csv saved")
    if _should_stop("summary", args.stop_after):
        return 0

    contributions_path = out_dir / "rbp_nucleotide_contributions.csv"
    if contributions_path.is_file():
        contributions = _load_result_csv(contributions_path, "cases")
    else:
        print("[RUN] cases")
        if effects.empty:
            contributions = pd.DataFrame()
        else:
            evaluator = RBPMotifMutagenesisEvaluator(
                get_model(),
                valid_pwms,
                prediction_scale=args.prediction_scale,
            )
            contributions = evaluator.compute_nucleotide_contributions(
                effects,
                samples,
                n_cases_per_direction=args.n_cases_per_direction,
                min_case_transcripts=args.min_transcripts,
                context_flank=args.context_flank,
                batch_size=args.batch_size,
                case_regions=args.case_regions,
                target_rbps=args.plot_case_rbp,
                target_regions=args.plot_case_region,
                target_hit_ids=args.plot_case_hit_id,
                target_transcript_ids=args.plot_case_transcript_id,
                target_motif_starts=args.plot_case_motif_start,
                targeted_cases_per_rbp=args.plot_cases_per_rbp,
                selection_mode=args.case_selection_mode,
            )
        _atomic_csv(contributions, contributions_path)
        print("[DONE] cases: rbp_nucleotide_contributions.csv saved")
    if _should_stop("cases", args.stop_after):
        return 0

    attribution_path = out_dir / "signed_translation_attribution_windows.csv"
    if attribution_path.is_file():
        attribution_windows = _load_result_csv(
            attribution_path, "attribution"
        )
    else:
        print("[RUN] attribution")
        if args.de_novo_source == "signed_attribution":
            attribution_windows = extract_signed_translation_attribution_windows(
                get_model(),
                samples,
                prediction_scale=args.prediction_scale,
                num_transcripts=args.de_novo_num_transcripts,
                peaks_per_direction=args.de_novo_peaks_per_direction,
                window_radius=args.context_flank,
                target_regions=args.de_novo_regions,
                random_state=args.random_state,
            )
        else:
            attribution_windows = pd.DataFrame()
        _atomic_csv(attribution_windows, attribution_path)
        print(
            "[DONE] attribution: "
            "signed_translation_attribution_windows.csv saved"
        )
    if _should_stop("attribution", args.stop_after):
        return 0

    de_novo_path = out_dir / "de_novo_translation_motifs.csv"
    alignments_path = out_dir / "de_novo_motif_alignments.json"
    if de_novo_path.is_file() and alignments_path.is_file():
        de_novo = _load_result_csv(de_novo_path, "de_novo")
        alignments = _load_result_json(alignments_path, "de_novo")
    else:
        print("[RUN] de_novo")
        source_table = (
            attribution_windows
            if args.de_novo_source == "signed_attribution"
            else effects
        )
        if source_table.empty:
            de_novo, alignments = pd.DataFrame(), {}
        else:
            de_novo, alignments = discover_de_novo_translation_motifs(
                source_table,
                sequence_col="Context_Sequence",
                effect_col=(
                    "Signed_Attribution"
                    if args.de_novo_source == "signed_attribution"
                    else "Delta_Log2_TE"
                ),
                unit_col="Tid",
                region_col="Region",
                peak_offset_col="Peak_Offset",
                discovery_regions=args.de_novo_regions,
                k_values=args.de_novo_k,
                extreme_quantile=args.de_novo_extreme_quantile,
                neutral_quantile=args.de_novo_neutral_quantile,
                min_foreground_occurrences=args.de_novo_min_occurrences,
                top_n_per_direction=args.de_novo_top_n_per_direction,
                logo_flank=args.de_novo_logo_flank,
            )
        _atomic_csv(de_novo, de_novo_path)
        _atomic_json(alignments, alignments_path)
        print("[DONE] de_novo: canonical CSV and JSON results saved")
    if _should_stop("de_novo", args.stop_after):
        return 0

    known_positions_path = out_dir / "known_rbp_position_profiles.csv"
    de_novo_positions_path = out_dir / "de_novo_motif_position_profiles.csv"
    if known_positions_path.is_file() and de_novo_positions_path.is_file():
        known_position_profiles = _load_result_csv(
            known_positions_path, "positions"
        )
        de_novo_position_profiles = _load_result_csv(
            de_novo_positions_path, "positions"
        )
    else:
        print("[RUN] positions")
        known_rbp_names = None
        if args.position_rbp_scope == "summary" and not summary.empty:
            known_rbp_names = summary["RBP_Name"].dropna().astype(str).unique()
        position_profiles = build_motif_position_profiles(
            samples,
            known_hits=hits,
            de_novo_motifs=de_novo,
            bin_size=args.position_bin_size,
            utr5_length=args.position_utr5_length,
            cds_length=args.position_cds_length,
            utr3_length=args.position_utr3_length,
            bins_per_region=args.position_bins_per_region,
            known_rbp_names=known_rbp_names,
            pseudocount=args.position_pseudocount,
        )
        known_position_profiles = position_profiles["known_rbp"]
        de_novo_position_profiles = position_profiles["de_novo"]
        _atomic_csv(known_position_profiles, known_positions_path)
        _atomic_csv(de_novo_position_profiles, de_novo_positions_path)
        print("[DONE] positions: canonical position-profile CSVs saved")
    if _should_stop("positions", args.stop_after):
        return 0

    print("[RUN] plots: checking canonical PDF outputs")
    plot_paths = []
    plot_notes = []
    if not args.skip_plots:
        from plot.rbp_scan import (
            plot_de_novo_translation_motif_logos,
            plot_motif_position_preference_heatmap,
            plot_rbp_nucleotide_contribution_cases,
            plot_rbp_translation_effect_summary,
            select_rbp_nucleotide_contribution_cases,
        )

        summary_pdf = out_dir / "rbp_translation_effect_summary.pdf"
        if summary_pdf.is_file():
            print(f"[SKIP] plot: loaded {summary_pdf.name}")
            plot_paths.append(str(summary_pdf))
        elif not summary.empty:
            try:
                plot_paths.append(plot_rbp_translation_effect_summary(
                    summary,
                    out_path=str(summary_pdf),
                    top_n_per_direction=args.plot_top_n_per_direction,
                    fdr_threshold=args.plot_fdr_threshold,
                ))
            except ValueError as error:
                plot_notes.append(f"RBP summary plot skipped: {error}")

        if not contributions.empty:
            try:
                selected_cases = select_rbp_nucleotide_contribution_cases(
                    contributions,
                    summary_df=summary,
                    cases_per_rbp=args.plot_cases_per_rbp,
                    target_rbps=args.plot_case_rbp,
                    target_regions=args.plot_case_region,
                    target_hit_ids=args.plot_case_hit_id,
                    target_transcript_ids=args.plot_case_transcript_id,
                    target_motif_starts=args.plot_case_motif_start,
                    max_cases=args.plot_max_cases,
                )
                cases_dir = out_dir / "cases"
                expected_case_paths = {}
                for row in selected_cases.itertuples(index=False):
                    expected = cases_dir / (
                        f"rbp_base_contribution.{row.RBP_Name}."
                        f"{row.Tid}.{row.Hit_ID}.pdf"
                    )
                    expected_case_paths[str(row.Hit_ID)] = expected
                missing_case_ids = [
                    hit_id for hit_id, path in expected_case_paths.items()
                    if not path.is_file()
                ]
                if missing_case_ids:
                    plot_rbp_nucleotide_contribution_cases(
                        contributions,
                        out_dir=str(cases_dir),
                        target_hit_ids=missing_case_ids,
                    )
                else:
                    print("[SKIP] plot: all selected case PDFs exist")
                plot_paths.extend(
                    str(path) for path in expected_case_paths.values()
                    if path.is_file()
                )
            except ValueError as error:
                plot_notes.append(f"RBP case plots skipped: {error}")

        logo_pdf = out_dir / "de_novo_translation_motif_logos.pdf"
        if logo_pdf.is_file():
            print(f"[SKIP] plot: loaded {logo_pdf.name}")
            plot_paths.append(str(logo_pdf))
        elif not de_novo.empty and alignments:
            try:
                plot_paths.append(plot_de_novo_translation_motif_logos(
                    de_novo,
                    alignments,
                    out_path=str(logo_pdf),
                    top_n_per_direction=args.plot_logo_top_n,
                ))
            except ValueError as error:
                plot_notes.append(f"De novo logo plot skipped: {error}")

        known_heatmap_pdf = (
            out_dir / "known_rbp_position_preference_heatmap.pdf"
        )
        if known_heatmap_pdf.is_file():
            print(f"[SKIP] plot: loaded {known_heatmap_pdf.name}")
            plot_paths.append(str(known_heatmap_pdf))
        elif not known_position_profiles.empty:
            try:
                plot_paths.append(plot_motif_position_preference_heatmap(
                    known_position_profiles,
                    out_path=str(known_heatmap_pdf),
                    cluster_mode=args.position_cluster_mode,
                    min_total_hits=args.position_min_hits,
                    max_features=args.position_max_features,
                    value_col="Log2_Positional_Enrichment",
                    width=args.position_heatmap_width,
                    row_height=args.position_row_height,
                    layout="combined",
                    vector_cells=True,
                ))
            except ValueError as error:
                plot_notes.append(
                    f"Known-RBP position heatmap skipped: {error}"
                )

        de_novo_heatmap_pdf = out_dir / "de_novo_position_preference_heatmap.pdf"
        if de_novo_heatmap_pdf.is_file():
            print(f"[SKIP] plot: loaded {de_novo_heatmap_pdf.name}")
            plot_paths.append(str(de_novo_heatmap_pdf))
        elif not de_novo_position_profiles.empty:
            try:
                plot_paths.append(plot_motif_position_preference_heatmap(
                    de_novo_position_profiles,
                    out_path=str(de_novo_heatmap_pdf),
                    cluster_mode=args.position_cluster_mode,
                    min_total_hits=args.position_min_hits,
                    max_features=args.position_max_features,
                    value_col="Log2_Positional_Enrichment",
                    width=args.position_heatmap_width,
                    row_height=args.position_row_height,
                    layout="regional_pages",
                ))
            except ValueError as error:
                plot_notes.append(
                    f"De novo position heatmap skipped: {error}"
                )
    else:
        plot_notes.append("Plotting disabled by --skip-plots.")
    plot_result = {
        "paths": [str(Path(path).resolve()) for path in plot_paths],
        "notes": plot_notes,
    }
    _atomic_json(plot_result, out_dir / "rbp_pipeline_plot_manifest.json")
    if _should_stop("plots", args.stop_after):
        return 0

    result_summary = {
        "n_samples": len(samples),
        "n_valid_pwms": len(valid_pwms),
        "n_rejected_pwms": int(
            (pwm_audit.get("Status", pd.Series(dtype=str)) != "Valid").sum()
        ),
        "n_hits": len(hits),
        "n_effects": len(effects),
        "n_summary_rows": len(summary),
        "n_contribution_rows": len(contributions),
        "n_attribution_windows": len(attribution_windows),
        "n_de_novo_motifs": len(de_novo),
        "n_known_position_profile_rows": len(known_position_profiles),
        "n_de_novo_position_profile_rows": len(de_novo_position_profiles),
        "pdf_outputs": plot_result.get("paths", []),
        "plot_notes": plot_result.get("notes", []),
    }
    _atomic_json(result_summary, out_dir / "rbp_pipeline_summary.json")
    print(json.dumps(result_summary, indent=2))
    print(f"Pipeline completed: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
