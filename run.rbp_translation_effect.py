#!/usr/bin/env python3
"""Run the resumable TRACE RBP translation-effect and de novo motif pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

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
    compute_known_motif_scan_signature,
    discover_de_novo_translation_motifs,
    extract_signed_translation_attribution_windows,
    load_known_motif_scan_cache,
    save_known_motif_scan_cache,
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


def _digest(payload: Any) -> str:
    """Create a stable SHA256 digest for a stage configuration."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_signature(path: Optional[str]) -> Optional[dict]:
    """Describe a file without hashing a potentially large model or dataset."""
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Input file was not found: {resolved}")
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _atomic_pickle(value: Any, path: Path) -> None:
    """Atomically serialize a Python checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _atomic_json(value: Any, path: Path) -> None:
    """Atomically serialize a JSON checkpoint or manifest."""
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


class StageCache:
    """Manage stage checkpoints with parameter-aware completion markers."""

    def __init__(
        self,
        out_dir: Path,
        resume: bool,
        force_from: Optional[str],
    ):
        self.root = out_dir / "checkpoints"
        self.root.mkdir(parents=True, exist_ok=True)
        self.resume = bool(resume)
        self.force_index = (
            len(STAGES) if force_from is None else STAGES.index(force_from)
        )
        self.stage_digests = {}

    def checkpoint_path(self, stage: str) -> Path:
        return self.root / f"{stage}.pkl"

    def marker_path(self, stage: str) -> Path:
        return self.root / f"{stage}.done.json"

    def stage_signature(
        self,
        stage: str,
        parameters: Mapping[str, Any],
        dependencies: Iterable[str] = (),
    ) -> str:
        payload = {
            "stage": stage,
            "parameters": dict(parameters),
            "dependencies": {
                dependency: self.stage_digests[dependency]
                for dependency in dependencies
            },
        }
        signature = _digest(payload)
        self.stage_digests[stage] = signature
        return signature

    def reusable(self, stage: str, signature: str) -> bool:
        if not self.resume or STAGES.index(stage) >= self.force_index:
            return False
        marker_path = self.marker_path(stage)
        checkpoint_path = self.checkpoint_path(stage)
        if not marker_path.is_file() or not checkpoint_path.is_file():
            return False
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return marker.get("signature") == signature

    def load(self, stage: str):
        with self.checkpoint_path(stage).open("rb") as handle:
            value = pickle.load(handle)
        print(f"[SKIP] {stage}: loaded completed checkpoint")
        return value

    def save(self, stage: str, signature: str, value: Any) -> None:
        _atomic_pickle(value, self.checkpoint_path(stage))
        _atomic_json(
            {
                "stage": stage,
                "signature": signature,
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "checkpoint": str(self.checkpoint_path(stage)),
            },
            self.marker_path(stage),
        )
        print(f"[DONE] {stage}: checkpoint saved")


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
            "Resumable known-RBP perturbation and de novo translation motif "
            "analysis for TRACE BaseModel checkpoints."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    required = parser.add_argument_group("required inputs")
    required.add_argument("--model-config", required=True)
    required.add_argument("--checkpoint", required=True)
    required.add_argument("--dataset", action="append", required=True)
    required.add_argument("--pwm-pkl", required=True)
    required.add_argument("--metadata-tsv", required=True)
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
        help="Worker threads used to scan independent transcripts for RBP motifs.",
    )
    selection.add_argument(
        "--known-motif-scan-cache-path",
        help=(
            "Portable known-RBP hit cache path; defaults to "
            "OUT_DIR/known_rbp_motif_hits.pkl."
        ),
    )
    selection.add_argument("--random-state", type=int, default=42)

    statistics = parser.add_argument_group("statistics and discovery")
    statistics.add_argument("--min-transcripts", type=int, default=5)
    statistics.add_argument("--bootstrap-iterations", type=int, default=2000)
    statistics.add_argument("--confidence-level", type=float, default=0.95)
    statistics.add_argument("--n-cases-per-direction", type=int, default=3)
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
    statistics.add_argument("--de-novo-logo-flank", type=int, default=3)

    plotting = parser.add_argument_group("PDF plotting")
    plotting.add_argument("--skip-plots", action="store_true")
    plotting.add_argument("--plot-top-n-per-direction", type=int, default=12)
    plotting.add_argument("--plot-fdr-threshold", type=float)
    plotting.add_argument("--plot-max-cases", type=int, default=6)
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
    plotting.add_argument("--position-min-hits", type=int, default=10)
    plotting.add_argument(
        "--position-max-features",
        type=int,
        default=80,
        help="Maximum heatmap rows; use 0 to show every retained feature.",
    )
    plotting.add_argument(
        "--position-rbp-scope",
        choices=["summary", "all"],
        default="summary",
        help="Use statistically summarized RBPs or every scanned RBP.",
    )
    plotting.add_argument("--position-pseudocount", type=float, default=0.5)
    plotting.add_argument("--position-heatmap-width", type=float, default=7.2)
    plotting.add_argument("--position-row-height", type=float, default=0.22)

    resume = parser.add_argument_group("checkpoint control")
    resume.add_argument("--no-resume", action="store_true")
    resume.add_argument("--force-from", choices=STAGES)
    resume.add_argument("--stop-after", choices=STAGES)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Execute all requested stages with resumable checkpoints."""
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = StageCache(
        out_dir,
        resume=not args.no_resume,
        force_from=args.force_from,
    )
    device = _device_from_argument(args.device)

    target_rbps = _parse_csv_values(args.target_rbp)
    rbps_from_file = _load_id_collection(args.target_rbp_file)
    if rbps_from_file:
        target_rbps = list(dict.fromkeys((target_rbps or []) + rbps_from_file))
    target_transcripts = _load_id_collection(args.target_transcript_file)

    input_signatures = {
        "model_config": _file_signature(args.model_config),
        "checkpoint": _file_signature(args.checkpoint),
        "datasets": [_file_signature(path) for path in args.dataset],
        "pwm_pkl": _file_signature(args.pwm_pkl),
        "metadata_tsv": _file_signature(args.metadata_tsv),
        "target_rbp_file": _file_signature(args.target_rbp_file),
        "target_transcript_file": _file_signature(args.target_transcript_file),
    }
    _atomic_json(
        {
            "arguments": vars(args),
            "input_signatures": input_signatures,
            "stages": list(STAGES),
        },
        out_dir / "rbp_pipeline_run_manifest.json",
    )

    pwm_library = _load_pickle(args.pwm_pkl)
    if not isinstance(pwm_library, dict):
        raise TypeError("--pwm-pkl must contain a dictionary keyed by Matrix_id.")
    metadata = pd.read_csv(args.metadata_tsv, sep="\t", dtype={"Matrix_id": str})

    model_holder = {}
    dataset_holder = {}

    def get_model():
        if "model" not in model_holder:
            model_holder["model"] = _load_model(args, device)
        return model_holder["model"]

    def get_dataset():
        if "dataset" not in dataset_holder:
            dataset_holder["dataset"] = _load_dataset(args.dataset)
        return dataset_holder["dataset"]

    validation_signature = cache.stage_signature(
        "validate_pwms",
        {
            "pwm": input_signatures["pwm_pkl"],
            "metadata": input_signatures["metadata_tsv"],
            "target_rbps": target_rbps,
        },
    )
    if cache.reusable("validate_pwms", validation_signature):
        validation_result = cache.load("validate_pwms")
    else:
        print("[RUN] validate_pwms")
        valid_pwms, pwm_audit = validate_rbp_pwm_library(
            pwm_library,
            metadata=metadata,
            target_rbps=target_rbps,
        )
        validation_result = {"valid_pwms": valid_pwms, "audit": pwm_audit}
        cache.save("validate_pwms", validation_signature, validation_result)
        _atomic_csv(pwm_audit, out_dir / "rbp_pwm_validation.csv")
    valid_pwms = validation_result["valid_pwms"]
    pwm_audit = validation_result["audit"]
    if not (out_dir / "rbp_pwm_validation.csv").is_file():
        _atomic_csv(pwm_audit, out_dir / "rbp_pwm_validation.csv")
    if _should_stop("validate_pwms", args.stop_after):
        return 0

    samples_signature = cache.stage_signature(
        "samples",
        {
            "datasets": input_signatures["datasets"],
            "target_transcripts": target_transcripts,
            "num_transcripts": args.num_transcripts,
            "random_state": args.random_state,
            "zero_expression": not args.use_dataset_expression,
        },
    )
    if cache.reusable("samples", samples_signature):
        samples = cache.load("samples")
    else:
        print("[RUN] samples")
        samples = collect_unique_transcript_samples(
            get_dataset(),
            target_transcript_ids=target_transcripts,
            num_transcripts=args.num_transcripts,
            random_state=args.random_state,
        )
        if not args.use_dataset_expression:
            for sample in samples.values():
                sample["Expr_Vector"] = np.zeros_like(sample["Expr_Vector"])
        cache.save("samples", samples_signature, samples)
    if _should_stop("samples", args.stop_after):
        return 0

    portable_hits_path = (
        Path(args.known_motif_scan_cache_path).expanduser().resolve()
        if args.known_motif_scan_cache_path
        else out_dir / "known_rbp_motif_hits.pkl"
    )
    portable_hits_signature = compute_known_motif_scan_signature(
        samples,
        valid_pwms,
        metadata,
        target_rbps=target_rbps,
        regions=args.regions,
        score_threshold=args.score_threshold,
        max_hits_per_rbp_transcript_region=(
            args.max_hits_per_rbp_transcript_region
        ),
        context_flank=args.context_flank,
    )
    hits_signature = cache.stage_signature(
        "hits",
        {
            "portable_scan_signature": portable_hits_signature,
        },
        dependencies=("validate_pwms", "samples"),
    )
    can_reuse_hits = (
        not args.no_resume
        and (
            args.force_from is None
            or STAGES.index("hits") < STAGES.index(args.force_from)
        )
    )
    hits = None
    if can_reuse_hits:
        hits = load_known_motif_scan_cache(
            str(portable_hits_path),
            expected_signature=portable_hits_signature,
        )
    if hits is None:
        print("[RUN] hits")
        hits = collect_rbp_motif_hits(
            samples,
            valid_pwms,
            metadata,
            target_rbps=target_rbps,
            regions=args.regions,
            score_threshold=args.score_threshold,
            max_hits_per_rbp_transcript_region=(
                args.max_hits_per_rbp_transcript_region
            ),
            context_flank=args.context_flank,
            num_workers=args.scan_workers,
        )
        save_known_motif_scan_cache(
            hits,
            str(portable_hits_path),
            signature=portable_hits_signature,
        )
        _atomic_csv(hits, out_dir / "rbp_motif_hits.csv")
        print("[DONE] hits: portable motif-position cache saved")
    else:
        print("[SKIP] hits: loaded portable motif-position cache")
    if not (out_dir / "rbp_motif_hits.csv").is_file():
        _atomic_csv(hits, out_dir / "rbp_motif_hits.csv")
    if _should_stop("hits", args.stop_after):
        return 0

    model_signature = {
        "model_config": input_signatures["model_config"],
        "checkpoint": input_signatures["checkpoint"],
        "head_hidden_dim": args.head_hidden_dim,
        "prediction_scale": args.prediction_scale,
        "non_strict": args.non_strict,
    }
    effects_signature = cache.stage_signature(
        "effects",
        {**model_signature, "batch_size": args.batch_size},
        dependencies=("hits", "samples", "validate_pwms"),
    )
    if cache.reusable("effects", effects_signature):
        effects = cache.load("effects")
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
        cache.save("effects", effects_signature, effects)
        _atomic_csv(effects, out_dir / "rbp_motif_hit_effects.csv")
    if not (out_dir / "rbp_motif_hit_effects.csv").is_file():
        _atomic_csv(effects, out_dir / "rbp_motif_hit_effects.csv")
    if _should_stop("effects", args.stop_after):
        return 0

    summary_signature = cache.stage_signature(
        "summary",
        {
            "min_transcripts": args.min_transcripts,
            "bootstrap_iterations": args.bootstrap_iterations,
            "confidence_level": args.confidence_level,
            "random_state": args.random_state,
        },
        dependencies=("effects",),
    )
    if cache.reusable("summary", summary_signature):
        summary = cache.load("summary")
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
        cache.save("summary", summary_signature, summary)
        _atomic_csv(summary, out_dir / "rbp_motif_effect_summary.csv")
    if not (out_dir / "rbp_motif_effect_summary.csv").is_file():
        _atomic_csv(summary, out_dir / "rbp_motif_effect_summary.csv")
    if _should_stop("summary", args.stop_after):
        return 0

    cases_signature = cache.stage_signature(
        "cases",
        {
            **model_signature,
            "n_cases_per_direction": args.n_cases_per_direction,
            "min_transcripts": args.min_transcripts,
            "context_flank": args.context_flank,
            "batch_size": args.batch_size,
        },
        dependencies=("effects", "samples", "validate_pwms"),
    )
    if cache.reusable("cases", cases_signature):
        contributions = cache.load("cases")
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
            )
        cache.save("cases", cases_signature, contributions)
        _atomic_csv(
            contributions,
            out_dir / "rbp_nucleotide_contributions.csv",
        )
    if not (out_dir / "rbp_nucleotide_contributions.csv").is_file():
        _atomic_csv(contributions, out_dir / "rbp_nucleotide_contributions.csv")
    if _should_stop("cases", args.stop_after):
        return 0

    attribution_signature = cache.stage_signature(
        "attribution",
        {
            **model_signature,
            "enabled": args.de_novo_source == "signed_attribution",
            "num_transcripts": args.de_novo_num_transcripts,
            "peaks_per_direction": args.de_novo_peaks_per_direction,
            "window_radius": args.context_flank,
            "random_state": args.random_state,
        },
        dependencies=("samples",),
    )
    if cache.reusable("attribution", attribution_signature):
        attribution_windows = cache.load("attribution")
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
                random_state=args.random_state,
            )
        else:
            attribution_windows = pd.DataFrame()
        cache.save("attribution", attribution_signature, attribution_windows)
        _atomic_csv(
            attribution_windows,
            out_dir / "signed_translation_attribution_windows.csv",
        )
    if not (out_dir / "signed_translation_attribution_windows.csv").is_file():
        _atomic_csv(
            attribution_windows,
            out_dir / "signed_translation_attribution_windows.csv",
        )
    if _should_stop("attribution", args.stop_after):
        return 0

    de_novo_dependency = (
        "attribution" if args.de_novo_source == "signed_attribution" else "effects"
    )
    de_novo_signature = cache.stage_signature(
        "de_novo",
        {
            "source": args.de_novo_source,
            "k_values": args.de_novo_k,
            "extreme_quantile": args.de_novo_extreme_quantile,
            "neutral_quantile": args.de_novo_neutral_quantile,
            "min_occurrences": args.de_novo_min_occurrences,
            "top_n": args.de_novo_top_n_per_direction,
            "logo_flank": args.de_novo_logo_flank,
        },
        dependencies=(de_novo_dependency,),
    )
    if cache.reusable("de_novo", de_novo_signature):
        de_novo_result = cache.load("de_novo")
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
                k_values=args.de_novo_k,
                extreme_quantile=args.de_novo_extreme_quantile,
                neutral_quantile=args.de_novo_neutral_quantile,
                min_foreground_occurrences=args.de_novo_min_occurrences,
                top_n_per_direction=args.de_novo_top_n_per_direction,
                logo_flank=args.de_novo_logo_flank,
            )
        de_novo_result = {"table": de_novo, "alignments": alignments}
        cache.save("de_novo", de_novo_signature, de_novo_result)
        _atomic_csv(de_novo, out_dir / "de_novo_translation_motifs.csv")
        _atomic_json(alignments, out_dir / "de_novo_motif_alignments.json")
    de_novo = de_novo_result["table"]
    alignments = de_novo_result["alignments"]
    if not (out_dir / "de_novo_translation_motifs.csv").is_file():
        _atomic_csv(de_novo, out_dir / "de_novo_translation_motifs.csv")
    if not (out_dir / "de_novo_motif_alignments.json").is_file():
        _atomic_json(alignments, out_dir / "de_novo_motif_alignments.json")
    if _should_stop("de_novo", args.stop_after):
        return 0

    positions_signature = cache.stage_signature(
        "positions",
        {
            "bin_size": args.position_bin_size,
            "utr5_length": args.position_utr5_length,
            "cds_length": args.position_cds_length,
            "utr3_length": args.position_utr3_length,
            "bins_per_region": args.position_bins_per_region,
            "rbp_scope": args.position_rbp_scope,
            "pseudocount": args.position_pseudocount,
        },
        dependencies=("samples", "hits", "summary", "de_novo"),
    )
    if cache.reusable("positions", positions_signature):
        position_profiles = cache.load("positions")
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
        cache.save("positions", positions_signature, position_profiles)
        _atomic_csv(
            position_profiles["known_rbp"],
            out_dir / "known_rbp_position_profiles.csv",
        )
        _atomic_csv(
            position_profiles["de_novo"],
            out_dir / "de_novo_motif_position_profiles.csv",
        )
    known_position_profiles = position_profiles["known_rbp"]
    de_novo_position_profiles = position_profiles["de_novo"]
    if not (out_dir / "known_rbp_position_profiles.csv").is_file():
        _atomic_csv(
            known_position_profiles,
            out_dir / "known_rbp_position_profiles.csv",
        )
    if not (out_dir / "de_novo_motif_position_profiles.csv").is_file():
        _atomic_csv(
            de_novo_position_profiles,
            out_dir / "de_novo_motif_position_profiles.csv",
        )
    if _should_stop("positions", args.stop_after):
        return 0

    plots_signature = cache.stage_signature(
        "plots",
        {
            "skip": args.skip_plots,
            "top_n": args.plot_top_n_per_direction,
            "fdr_threshold": args.plot_fdr_threshold,
            "max_cases": args.plot_max_cases,
            "logo_top_n": args.plot_logo_top_n,
            "position_cluster_mode": args.position_cluster_mode,
            "position_min_hits": args.position_min_hits,
            "position_max_features": args.position_max_features,
            "position_heatmap_width": args.position_heatmap_width,
            "position_row_height": args.position_row_height,
            "format": "pdf_only",
        },
        dependencies=("summary", "cases", "de_novo", "positions"),
    )
    reuse_plots = cache.reusable("plots", plots_signature)
    if reuse_plots:
        plot_result = cache.load("plots")
        existing_paths = [Path(path) for path in plot_result.get("paths", [])]
        if not all(path.is_file() for path in existing_paths):
            print("[RERUN] plots: one or more recorded PDFs are missing")
            reuse_plots = False
    if not reuse_plots:
        print("[RUN] plots")
        plot_paths = []
        plot_notes = []
        if not args.skip_plots:
            # Import plotting dependencies only when PDF generation is requested.
            from plot.rbp_scan import (
                plot_de_novo_translation_motif_logos,
                plot_motif_position_preference_heatmap,
                plot_rbp_nucleotide_contribution_cases,
                plot_rbp_translation_effect_summary,
            )

            if not summary.empty:
                try:
                    plot_paths.append(plot_rbp_translation_effect_summary(
                        summary,
                        out_path=str(out_dir / "rbp_translation_effect_summary.pdf"),
                        top_n_per_direction=args.plot_top_n_per_direction,
                        fdr_threshold=args.plot_fdr_threshold,
                    ))
                except ValueError as error:
                    plot_notes.append(f"RBP summary plot skipped: {error}")
            if not contributions.empty:
                plot_paths.extend(plot_rbp_nucleotide_contribution_cases(
                    contributions,
                    out_dir=str(out_dir / "cases"),
                    max_cases=args.plot_max_cases,
                ))
            if not de_novo.empty and alignments:
                try:
                    plot_paths.append(plot_de_novo_translation_motif_logos(
                        de_novo,
                        alignments,
                        out_path=str(
                            out_dir / "de_novo_translation_motif_logos.pdf"
                        ),
                        top_n_per_direction=args.plot_logo_top_n,
                    ))
                except ValueError as error:
                    plot_notes.append(f"De novo logo plot skipped: {error}")
            if not known_position_profiles.empty:
                try:
                    plot_paths.append(plot_motif_position_preference_heatmap(
                        known_position_profiles,
                        out_path=str(
                            out_dir / "known_rbp_position_preference_heatmap.pdf"
                        ),
                        cluster_mode=args.position_cluster_mode,
                        min_total_hits=args.position_min_hits,
                        max_features=args.position_max_features,
                        width=args.position_heatmap_width,
                        row_height=args.position_row_height,
                    ))
                except ValueError as error:
                    plot_notes.append(f"Known-RBP position heatmap skipped: {error}")
            if not de_novo_position_profiles.empty:
                try:
                    plot_paths.append(plot_motif_position_preference_heatmap(
                        de_novo_position_profiles,
                        out_path=str(
                            out_dir / "de_novo_position_preference_heatmap.pdf"
                        ),
                        cluster_mode=args.position_cluster_mode,
                        min_total_hits=args.position_min_hits,
                        max_features=args.position_max_features,
                        width=args.position_heatmap_width,
                        row_height=args.position_row_height,
                    ))
                except ValueError as error:
                    plot_notes.append(f"De novo position heatmap skipped: {error}")
        else:
            plot_notes.append("Plotting disabled by --skip-plots.")
        plot_result = {
            "paths": [str(Path(path).resolve()) for path in plot_paths],
            "notes": plot_notes,
        }
        cache.save("plots", plots_signature, plot_result)
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
