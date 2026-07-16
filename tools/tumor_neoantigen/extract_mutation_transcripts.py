#!/usr/bin/env python3
"""
Extract transcripts overlapping somatic variants for TRACE prediction.

Reads a VCF to collect all PASS variant positions, then queries a GTF
to find every transcript whose exons overlap at least one variant.
Outputs a CSV of unique transcript IDs suitable as input to run_trace_prediction.py
(--input_csv).
"""
import os, re, sys, gzip, argparse
import pandas as pd
from collections import defaultdict

def main():
    p = argparse.ArgumentParser(
        description="Extract transcripts overlapping VCF variants for TRACE input.")
    p.add_argument("--vcf", required=True, help="Input VCF (.vcf or .vcf.gz)")
    p.add_argument("--gtf", required=True, help="Reference GTF annotation")
    p.add_argument("--output", required=True, help="Output CSV (Transcript_ID, Tumor_Run)")
    p.add_argument("--tumor_run", default="", help="Tumor Run ID to populate Tumor_Run column")
    args = p.parse_args()

    # 1. Collect all PASS variant positions from VCF
    print("--- Collecting variant positions ---")
    positions = defaultdict(set)  # chrom -> set of positions
    vcf_open = gzip.open if args.vcf.endswith('.gz') else open
    n = 0
    with vcf_open(args.vcf, 'rt') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split('\t')
            if len(parts) < 7: continue
            if parts[6] != 'PASS': continue
            chrom = parts[0]; pos = int(parts[1])
            positions[chrom].add(pos)
            n += 1
    print(f" -> {n} PASS variants across {len(positions)} chromosomes")

    if n == 0:
        print("[Warning] No PASS variants found.")
        pd.DataFrame(columns=['Transcript_ID','Tumor_Run']).to_csv(args.output, index=False)
        return

    # 2. Scan GTF exons for overlapping transcripts
    print("--- Scanning GTF for overlapping transcripts ---")
    tx_re = re.compile(r'transcript_id "([^"]+)"')
    found_tx = set()
    opener = gzip.open if args.gtf.endswith('.gz') else open

    with opener(args.gtf, 'rt') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split('\t')
            if len(parts) < 9 or parts[2] != 'exon': continue
            chrom = parts[0]
            if chrom not in positions: continue
            start = int(parts[3]); end = int(parts[4])
            tx_match = tx_re.search(parts[8])
            if not tx_match: continue
            tx_id = tx_match.group(1).strip()

            # Check overlap with any variant position
            chrom_positions = positions[chrom]
            for pos in chrom_positions:
                if start <= pos <= end:
                    found_tx.add(tx_id)
                    break  # transcript found, no need to check more positions

    print(f" -> {len(found_tx)} unique transcripts overlap variants")

    # 3. Write output
    df = pd.DataFrame({
        'Transcript_ID': sorted(found_tx),
        'Tumor_Run': args.tumor_run
    })
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f" -> Saved: {args.output}")

if __name__ == "__main__":
    main()
