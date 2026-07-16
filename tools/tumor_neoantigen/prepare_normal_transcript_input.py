#!/usr/bin/env python3
"""
Generate a CSV of transcripts expressed in the normal sample for TRACE input.

Filters the TPM matrix to select transcripts with normal-sample TPM above a
threshold, then writes a minimal CSV (Transcript_ID, Tumor_Run) suitable as
the --input_csv argument of run_trace_prediction.py.

Usage:
    python prepare_normal_transcript_input.py \
        --tpm_csv transcript_tpm_matrix.csv \
        --normal_run SRR123456 \
        --output normal_expressed_transcripts.csv \
        --min_tpm 0.5
"""
import os, sys, argparse, pandas as pd

def main():
    p = argparse.ArgumentParser(
        description="Generate normal-expressed transcript CSV for TRACE input.")
    p.add_argument("--tpm_csv", required=True, help="Transcript TPM matrix CSV")
    p.add_argument("--normal_run", required=True, help="Normal sample Run ID")
    p.add_argument("--output", required=True, help="Output CSV")
    p.add_argument("--min_tpm", type=float, default=0.5,
                   help="Minimum TPM threshold (default: 0.5)")
    args = p.parse_args()

    if not os.path.exists(args.tpm_csv):
        print(f"[Error] TPM matrix not found: {args.tpm_csv}", file=sys.stderr)
        sys.exit(1)

    tpm = pd.read_csv(args.tpm_csv, index_col=0)

    if args.normal_run not in tpm.columns:
        print(f"[Error] Normal Run '{args.normal_run}' not found in TPM matrix columns",
              file=sys.stderr)
        sys.exit(1)

    norm = tpm[args.normal_run]
    expressed = norm[norm > args.min_tpm]

    if len(expressed) == 0:
        print(f"[Warning] No transcripts with TPM > {args.min_tpm} for {args.normal_run}",
              file=sys.stderr)
        # Still write an empty CSV so downstream checks can detect it
        pd.DataFrame(columns=['Transcript_ID', 'Tumor_Run']).to_csv(args.output, index=False)
        sys.exit(0)

    df = pd.DataFrame({
        'Transcript_ID': expressed.index,
        'Tumor_Run': args.normal_run
    })
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f" -> {len(df)} transcripts with TPM > {args.min_tpm} written to {args.output}")

if __name__ == "__main__":
    main()
