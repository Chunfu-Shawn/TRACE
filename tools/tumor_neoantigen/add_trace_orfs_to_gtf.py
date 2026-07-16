#!/usr/bin/env python3
"""
Build an enhanced GTF with TRACE-predicted CDS + reference canonical CDS.

Strategy:
  1. Parse TRACE ORF CSV -> generate CDS lines (TRACE-predicted ORFs take priority).
  2. Parse reference GTF -> extract CDS lines for canonical transcripts.
  3. For transcripts already covered by TRACE, skip the reference CDS entries.
  4. Merge: TRACE CDS lines (base) + ref CDS lines for uncovered transcripts.

This ensures every transcript with coding potential (from TRACE or canonical
annotation) has CDS coordinates, and TRACE predictions override reference
annotations when both exist.
"""
import os, sys, re, argparse, pandas as pd
from collections import defaultdict

def safe_clean_id(tid):
    tid_str = str(tid).strip()
    if tid_str.startswith('ENS'): return tid_str.split('.')[0]
    return tid_str

def main():
    p = argparse.ArgumentParser(
        description="Build enhanced GTF: TRACE CDS base + ref canonical CDS supplement.")
    p.add_argument("--trace_orf_csv", required=True,
                   help="TRACE ORF CSV (high_confidence_orfs.*.csv)")
    p.add_argument("--ref_gtf", required=True, help="Reference GTF")
    p.add_argument("--output_gtf", required=True, help="Output enhanced GTF")
    p.add_argument("--source_tag", default="TRACE",
                   help="GTF source column value for TRACE CDS lines (default: TRACE)")
    args = p.parse_args()

    # ------------------------------------------------------------------
    # 1. Build gene info lookup from reference GTF
    # ------------------------------------------------------------------
    print("--- Extracting gene metadata from reference GTF ---")
    tx_re = re.compile(r'transcript_id "([^"]+)"')
    gene_re = re.compile(r'gene_id "([^"]+)"')
    gname_re = re.compile(r'gene_name "([^"]+)"')

    tx_info = {}          # transcript_id -> {chr, strand, gene_id, gene_name}
    ref_cds_lines = []    # CDS lines from reference GTF
    tx_has_ref_cds = set()  # transcripts that have CDS in reference

    with open(args.ref_gtf) as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split('\t')
            if len(parts) < 9: continue
            feat = parts[2]

            info = parts[8]
            tx_match = tx_re.search(info)
            if not tx_match: continue
            tx_id = tx_match.group(1).strip()

            # Capture metadata (once per transcript)
            if tx_id not in tx_info:
                gm = gene_re.search(info)
                gn = gname_re.search(info)
                tx_info[tx_id] = {
                    'chr': parts[0],
                    'strand': parts[6],
                    'gene_id': gm.group(1).strip() if gm else 'Unknown',
                    'gene_name': gn.group(1).strip() if gn else 'Unknown',
                }

            # Collect CDS lines
            if feat == 'CDS':
                ref_cds_lines.append(line)
                tx_has_ref_cds.add(tx_id)

    print(f" -> {len(tx_info)} transcripts in reference GTF")
    print(f" -> {len(tx_has_ref_cds)} have canonical CDS")

    # ------------------------------------------------------------------
    # 2. Parse TRACE ORFs and build CDS lines
    # ------------------------------------------------------------------
    print("--- Loading TRACE ORF predictions ---")
    orf_df = pd.read_csv(args.trace_orf_csv)
    print(f" -> {len(orf_df)} ORF entries")

    trace_cds_lines = []
    trace_tx_set = set()
    seen = set()  # dedup key: (tid, start, stop)

    for _, row in orf_df.iterrows():
        tid = str(row.get('Tid', '')).strip()
        if not tid: continue
        start = int(row.get('start', 0))
        stop = int(row.get('stop', 0))
        if start <= 0 or stop <= 0: continue
        strand = str(row.get('strand', '+')).strip()
        if strand not in ('+', '-'): strand = '+'

        key = (tid, start, stop)
        if key in seen: continue
        seen.add(key)

        # Get gene metadata: try ref GTF, fall back to defaults
        info = tx_info.get(tid) or tx_info.get(safe_clean_id(tid)) or {}
        chrom = info.get('chr', 'TRACE_chr')
        strand_gtf = info.get('strand', strand)
        gene_id = info.get('gene_id', 'TRACE_gene')
        gene_name = info.get('gene_name', 'TRACE_gene')

        # Write a single CDS line for this ORF (phase=0)
        attr = (f'gene_id "{gene_id}"; transcript_id "{tid}"; '
                f'gene_name "{gene_name}";')
        cds_line = f'{chrom}\t{args.source_tag}\tCDS\t{start}\t{stop}\t.\t{strand_gtf}\t0\t{attr}\n'
        trace_cds_lines.append(cds_line)
        trace_tx_set.add(tid)
        trace_tx_set.add(safe_clean_id(tid))

    print(f" -> {len(trace_cds_lines)} TRACE CDS lines for {len(trace_tx_set)} unique transcripts")

    # ------------------------------------------------------------------
    # 3. Assemble final GTF: TRACE CDS (base) + ref CDS for uncovered tx
    # ------------------------------------------------------------------
    print("--- Writing enhanced GTF ---")
    n_ref_appended = 0

    with open(args.output_gtf, 'w') as fout:
        # Header comment
        fout.write(f"# Enhanced GTF: TRACE CDS base + ref canonical CDS supplement\n")
        fout.write(f"# TRACE ORFs: {len(trace_cds_lines)} lines\n")
        fout.write(f"# Reference CDS appended for transcripts without TRACE prediction\n")

        # TRACE CDS lines first
        for line in trace_cds_lines:
            fout.write(line)

        # Then reference CDS lines for transcripts NOT covered by TRACE
        for line in ref_cds_lines:
            # Check if this CDS line's transcript already has TRACE ORFs
            tx_match = tx_re.search(line)
            if tx_match:
                tx_id = tx_match.group(1).strip()
                if tx_id in trace_tx_set or safe_clean_id(tx_id) in trace_tx_set:
                    continue  # skip, TRACE already covers this transcript
            fout.write(line)
            n_ref_appended += 1

    print(f" -> Appended {n_ref_appended} reference CDS lines for uncovered transcripts")
    print(f" -> Enhanced GTF saved: {args.output_gtf}")
    print(f" -> Total CDS coverage: TRACE={len(trace_tx_set)} + ref supplement={n_ref_appended}")

if __name__ == "__main__":
    main()
