#!/usr/bin/env python3
"""Run TRACE for a tumor cohort while loading shared resources only once."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cohort-level TRACE prediction with one model/FASTA load."
    )
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--fasta_files", nargs="+", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--weights_path", required=True)
    expr_group = parser.add_mutually_exclusive_group(required=True)
    expr_group.add_argument("--expr_dict_path")
    expr_group.add_argument("--patient_counts_file")
    parser.add_argument("--counts_level", choices=("transcript", "gene"), default="gene")
    parser.add_argument("--ref_order")
    parser.add_argument("--mapping_json")
    parser.add_argument("--tx2gene_mapping")
    parser.add_argument("--tpm_csv")
    parser.add_argument("--tpm_level", choices=("transcript", "gene"), default="transcript")
    parser.add_argument("--mode", default="short")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min_len", type=int, default=200)
    parser.add_argument("--max_len", type=int, default=10000)
    parser.add_argument("--patient_ids", nargs="+", help="Optional patient subset for benchmarking.")
    parser.add_argument("--max_patients", type=int, help="Optional first-N patient limit for benchmarking.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")


def clean_id(value: object) -> str:
    """Match TRACE identifier cleaning without importing the model stack."""
    identifier = str(value).strip().split("|", 1)[0]
    if identifier.startswith("ENS"):
        return identifier.split(".")[0]
    return identifier.split(":")[0]


def build_clean_sequence_dict(sequence_dict: dict[str, str]) -> dict[str, str]:
    """Normalize predictor FASTA keys for direct reuse by the ORF caller."""
    cleaned = {}
    conflicting_ids = []
    for raw_id, sequence in sequence_dict.items():
        transcript_id = clean_id(raw_id)
        previous = cleaned.get(transcript_id)
        if previous is not None and previous != sequence:
            conflicting_ids.append(transcript_id)
            continue
        cleaned[transcript_id] = sequence.replace("U", "T")
    if conflicting_ids:
        examples = ", ".join(sorted(set(conflicting_ids))[:5])
        raise ValueError(
            "FASTA normalization produced transcript IDs with conflicting sequences. "
            f"Examples: {examples}"
        )
    return cleaned


def load_tx2gene(path: Optional[str]) -> dict[str, str]:
    if not path:
        return {}
    frame = pd.read_csv(path, sep="\t")
    preferred = ("Transcript stable ID", "Gene stable ID")
    if set(preferred).issubset(frame.columns):
        transcript_col, gene_col = preferred
    elif len(frame.columns) >= 2:
        transcript_col, gene_col = frame.columns[:2]
    else:
        raise ValueError("Transcript-to-gene mapping must contain at least two columns.")
    return {
        clean_id(transcript): clean_id(gene)
        for transcript, gene in zip(frame[transcript_col], frame[gene_col])
        if pd.notna(transcript) and pd.notna(gene)
    }


def load_tpm_matrix(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.to_series().astype(str).map(clean_id)
    return frame.groupby(frame.index).mean(numeric_only=True)


def main() -> int:
    args = parse_args()
    import torch

    from model.generate_cell_env_expr_array import generate_cell_env_expr_dict
    from model.orf_caller import TranslationSignalORFCaller
    from model.prediction_heads import PsiteDensityHead
    from model.base_model import BaseModel
    from model.translation_predictor import TranslationProfilePredictor

    targets = pd.read_csv(args.input_csv)
    require_columns(targets, {"Patient", "Tumor_Run", "Transcript_ID"}, "Input CSV")
    targets = targets.dropna(subset=["Patient", "Tumor_Run", "Transcript_ID"]).copy()
    targets["Patient"] = targets["Patient"].astype(str).str.strip().str.replace(r"\s+", "_", regex=True)
    targets["Tumor_Run"] = targets["Tumor_Run"].astype(str).str.strip()
    targets["Transcript_ID"] = targets["Transcript_ID"].map(clean_id)
    if args.patient_ids:
        requested_patients = {
            "_".join(str(patient).strip().split()) for patient in args.patient_ids
        }
        targets = targets[targets["Patient"].isin(requested_patients)]
    if args.max_patients is not None:
        if args.max_patients < 1:
            raise ValueError("--max_patients must be positive.")
        first_patients = targets["Patient"].drop_duplicates().head(args.max_patients)
        targets = targets[targets["Patient"].isin(first_patients)]
    if targets.empty:
        raise ValueError("No candidate rows remain after applying the patient subset.")

    run_counts = targets.groupby("Patient")["Tumor_Run"].nunique()
    ambiguous = run_counts[run_counts != 1]
    if not ambiguous.empty:
        raise ValueError(
            "Each patient must map to exactly one tumor run; ambiguous patients: "
            + ", ".join(ambiguous.index.astype(str))
        )

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    status_path = out_root / "trace_cohort_status.csv"
    previous_status = {}
    if status_path.exists():
        previous_frame = pd.read_csv(status_path)
        if "Patient" in previous_frame.columns:
            previous_status = previous_frame.set_index("Patient").to_dict("index")

    if args.patient_counts_file:
        required = {
            "ref_order": args.ref_order,
            "mapping_json": args.mapping_json,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"Missing expression-builder arguments: {missing}")
        expr_dict = generate_cell_env_expr_dict(
            counts_file=args.patient_counts_file,
            ref_order_path=args.ref_order,
            mapping_json_path=args.mapping_json,
            quant_level=args.counts_level,
            tx2gene_file=args.tx2gene_mapping,
            min_tpm_threshold=0.0,
        )
    else:
        expr_dict = torch.load(args.expr_dict_path, map_location="cpu")

    tx2gene_dict = load_tx2gene(args.tx2gene_mapping)
    tpm_matrix = load_tpm_matrix(args.tpm_csv)

    model = BaseModel.from_config(args.config_path).to(args.device)
    model.add_head(
        "count",
        PsiteDensityHead.create_from_model(model, d_pred_h=384),
        overwrite=True,
    )
    model.load_pretrained_weights(args.weights_path, strict=False)
    predictor = TranslationProfilePredictor(model=model, fasta_files=args.fasta_files)
    orf_sequence_dict = build_clean_sequence_dict(predictor.seq_dict)
    print(
        "[FASTA] Reusable ORF sequence dictionary: "
        f"{len(orf_sequence_dict)} normalized transcript IDs."
    )

    status_rows = []
    for patient, patient_frame in targets.groupby("Patient", sort=False):
        patient_started = time.perf_counter()
        tumor_run = patient_frame["Tumor_Run"].iloc[0]
        patient_dir = out_root / patient
        patient_dir.mkdir(parents=True, exist_ok=True)
        protein_path = patient_dir / f"high_confidence_proteins.{patient}.{args.mode}_mode.fasta"
        orf_path = patient_dir / f"high_confidence_orfs.{patient}.{args.mode}_mode.csv"
        # The versioned marker prevents reuse of outputs produced before FASTA-ID normalization.
        complete_marker = patient_dir / f".trace_complete.id_normalization_v2.{args.mode}"

        if not args.overwrite and complete_marker.exists():
            print(f"[Skip] Completed TRACE output: {patient}")
            status_rows.append(
                {
                    "Patient": patient,
                    "Tumor_Run": tumor_run,
                    "Status": "skipped_complete",
                    "Elapsed_Seconds": previous_status.get(patient, {}).get(
                        "Elapsed_Seconds", 0.0
                    ),
                }
            )
            continue
        if tumor_run not in expr_dict:
            print(f"[Warning] Expression vector is missing for {patient}: {tumor_run}")
            status_rows.append(
                {
                    "Patient": patient,
                    "Tumor_Run": tumor_run,
                    "Status": "missing_expression",
                    "Elapsed_Seconds": 0.0,
                }
            )
            continue

        # Move pre-fix products aside before recomputing so stale MSTRG-only files cannot be reused.
        for stale_output in (protein_path, orf_path):
            if stale_output.exists():
                backup_output = stale_output.with_name(stale_output.name + ".pre_id_fix")
                if not backup_output.exists():
                    stale_output.rename(backup_output)

        transcript_ids = patient_frame["Transcript_ID"].drop_duplicates().tolist()
        matched_transcript_ids = [
            transcript_id
            for transcript_id in transcript_ids
            if transcript_id in orf_sequence_dict
        ]
        missing_transcript_ids = sorted(set(transcript_ids) - set(matched_transcript_ids))
        target_enst_count = sum(transcript_id.startswith("ENST") for transcript_id in transcript_ids)
        matched_enst_count = sum(
            transcript_id.startswith("ENST") for transcript_id in matched_transcript_ids
        )
        print(
            f"[TRACE] {patient}: targets={len(transcript_ids)}, "
            f"FASTA-matched={len(matched_transcript_ids)}, "
            f"ENST={matched_enst_count}/{target_enst_count}"
        )
        if missing_transcript_ids:
            print(
                f"[Warning] {patient}: {len(missing_transcript_ids)} target transcripts "
                "were not found in the normalized FASTA dictionary. "
                f"Examples: {', '.join(missing_transcript_ids[:5])}"
            )
        pkl_path = predictor.run(
            species="human",
            cell_type=patient,
            cell_expr_vector=expr_dict[tumor_run],
            target_tids=transcript_ids,
            out_dir=str(patient_dir),
            suffix=patient,
            min_len=args.min_len,
            max_len=args.max_len,
            batch_size=args.batch_size,
        )
        if not pkl_path:
            status_rows.append(
                {
                    "Patient": patient,
                    "Tumor_Run": tumor_run,
                    "Status": "no_fasta_match",
                    "Elapsed_Seconds": time.perf_counter() - patient_started,
                }
            )
            continue

        tpm_dict = None
        if tpm_matrix is not None:
            if tumor_run in tpm_matrix.columns:
                tpm_dict = tpm_matrix[tumor_run].to_dict()
            else:
                print(f"[Warning] TPM column is missing for {patient}: {tumor_run}")

        caller = TranslationSignalORFCaller(
            fasta_files=args.fasta_files,
            pkl_file=pkl_path,
            cell_type=patient,
            tpm_level=args.tpm_level,
            seq_dict=orf_sequence_dict,
            tx2gene_dict=tx2gene_dict,
            tpm_dict=tpm_dict,
        )
        predicted_ids = set(caller.preds_data[patient])
        predicted_enst_count = sum(
            transcript_id.startswith("ENST") for transcript_id in predicted_ids
        )
        if matched_enst_count > 0 and predicted_enst_count == 0:
            raise RuntimeError(
                f"{patient}: {matched_enst_count} ENST targets matched FASTA, but no ENST "
                "prediction reached the ORF caller. Check transcript-ID normalization."
            )
        orfs = caller.run(
            out_dir=str(patient_dir),
            start_codons=["ATG", "CTG", "GTG", "TTG", "ACG"],
            min_len=30,
            mode=args.mode,
            use_mane_filter=False,
            plot_density=False,
            hard_thresh_intensity=0,
            hard_thresh_periodicity=0.5,
            hard_thresh_uniformity=0.3,
            hard_thresh_step_up=0.51,
            hard_thresh_drop_off=0.51,
        )
        status_rows.append(
            {
                "Patient": patient,
                "Tumor_Run": tumor_run,
                "Status": "complete" if orf_path.exists() else "no_orfs",
                "Transcripts": len(transcript_ids),
                "FASTA_Matched_Transcripts": len(matched_transcript_ids),
                "Target_ENST": target_enst_count,
                "FASTA_Matched_ENST": matched_enst_count,
                "Predicted_Transcripts": len(predicted_ids),
                "Predicted_ENST": predicted_enst_count,
                "ORFs": len(orfs),
                "ORF_ENST": int(orfs["Tid"].astype(str).str.startswith("ENST").sum())
                if not orfs.empty and "Tid" in orfs.columns
                else 0,
                "Elapsed_Seconds": time.perf_counter() - patient_started,
            }
        )
        complete_marker.touch()
        pd.DataFrame(status_rows).to_csv(status_path, index=False)

    pd.DataFrame(status_rows).to_csv(status_path, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
