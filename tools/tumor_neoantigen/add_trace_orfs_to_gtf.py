#!/usr/bin/env python3
"""
Build an enhanced GTF with TRACE-predicted CDS + reference canonical CDS.

Strategy:
  1. Parse TRACE ORF CSV -> convert transcript-relative ORF coordinates to
     genomic coordinates using exon structures from reference (and optional
     extra) GTFs.  Write CDS lines with correct genomic positions and strand.
  2. Parse reference GTF -> extract canonical CDS lines.
  3. For transcripts already covered by TRACE, skip reference CDS entries.
  4. Merge: TRACE CDS (base) + ref CDS for uncovered transcripts.
"""
import os, sys, re, argparse, pandas as pd
from collections import defaultdict

def safe_clean_id(tid):
    tid_str = str(tid).strip()
    if tid_str.startswith('ENS'): return tid_str.split('.')[0]
    return tid_str

def parse_gtf_exons(gtf_path):
    """
    Parse exon lines from a GTF to build transcript structure maps.

    Returns: {transcript_id: {chr, strand, gene_id, gene_name,
                              exons: [(start, end), ...] sorted by genomic pos}}
    """
    tx_re = re.compile(r'transcript_id "([^"]+)"')
    gene_re = re.compile(r'gene_id "([^"]+)"')
    gname_re = re.compile(r'gene_name "([^"]+)"')

    tx_exons = defaultdict(list)
    tx_info = {}

    with open(gtf_path) as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split('\t')
            if len(parts) < 9: continue
            feat = parts[2]
            if feat not in ('exon', 'CDS', 'transcript'): continue

            info = parts[8]
            tx_match = tx_re.search(info)
            if not tx_match: continue
            tid = tx_match.group(1).strip()

            if tid not in tx_info:
                gm = gene_re.search(info)
                gn = gname_re.search(info)
                tx_info[tid] = {
                    'chr': parts[0],
                    'strand': parts[6],
                    'gene_id': gm.group(1).strip() if gm else 'Unknown',
                    'gene_name': gn.group(1).strip() if gn else 'Unknown',
                }

            if feat == 'exon':
                tx_exons[tid].append((int(parts[3]), int(parts[4])))

    # Sort exons and compute transcript length
    for tid in tx_exons:
        tx_exons[tid].sort(key=lambda x: x[0])
        tx_info[tid]['exons'] = tx_exons[tid]
        tx_info[tid]['tx_len'] = sum(e[1] - e[0] + 1 for e in tx_exons[tid])

    return tx_info

def build_pos_map(tx_info):
    """
    Build transcript-position -> genomic-position mapping for each transcript.

    Transcript position 1 = first base of the transcript (5' end).
    For + strand: position 1 = first exon start.
    For - strand: position 1 = last exon end (going backwards).
    """
    pos_maps = {}
    for tid, info in tx_info.items():
        if 'exons' not in info: continue
        exons = info['exons']
        strand = info['strand']
        pos_map = {}
        tx_pos = 1
        if strand == '+':
            for start, end in exons:
                for gp in range(start, end + 1):
                    pos_map[tx_pos] = gp
                    tx_pos += 1
        else:
            for start, end in reversed(exons):
                for gp in range(end, start - 1, -1):
                    pos_map[tx_pos] = gp
                    tx_pos += 1
        pos_maps[tid] = pos_map
    return pos_maps

def main():
    p = argparse.ArgumentParser(
        description="Build enhanced GTF: TRACE CDS (genomic coords) + ref canonical CDS.")
    p.add_argument("--trace_orf_csv", required=True,
                   help="TRACE ORF CSV (high_confidence_orfs.*.csv)")
    p.add_argument("--ref_gtf", required=True, help="Reference GTF")
    p.add_argument("--extra_gtf", nargs='*', default=[],
                   help="Additional GTFs for novel transcript exon structures")
    p.add_argument("--output_gtf", required=True, help="Output enhanced GTF")
    p.add_argument("--source_tag", default="TRACE",
                   help="GTF source column for TRACE CDS lines (default: TRACE)")
    args = p.parse_args()

    # ------------------------------------------------------------------
    # 1. Build transcript structure from all GTFs
    # ------------------------------------------------------------------
    print("--- Parsing transcript exon structures ---")
    tx_structure = parse_gtf_exons(args.ref_gtf)
    for gtf in args.extra_gtf:
        if os.path.exists(gtf):
            extra = parse_gtf_exons(gtf)
            for tid, info in extra.items():
                if tid not in tx_structure:
                    tx_structure[tid] = info
    print(f" -> {len(tx_structure)} transcripts with exon info")

    # ------------------------------------------------------------------
    # 2. Build position maps for coordinate conversion
    # ------------------------------------------------------------------
    print("--- Building transcript -> genomic position maps ---")
    pos_maps = build_pos_map(tx_structure)
    n_mapped = sum(1 for v in pos_maps.values() if v)
    print(f" -> {n_mapped} transcripts with position maps")

    # ------------------------------------------------------------------
    # 3. Load TRACE ORFs and convert to genomic coordinates
    # ------------------------------------------------------------------
    print("\n--- Loading TRACE ORFs and converting to genomic coordinates ---")
    orf_df = pd.read_csv(args.trace_orf_csv)
    print(f" -> {len(orf_df)} ORF entries")

    trace_cds_lines = []
    trace_tx_set = set()
    seen = set()
    n_skipped = 0

    for _, row in orf_df.iterrows():
        tid = str(row.get('Tid', '')).strip()
        if not tid: continue
        orf_start = int(row.get('start', 0))
        orf_stop = int(row.get('stop', 0))
        if orf_start <= 0 or orf_stop <= 0: continue

        # Look up transcript structure (try versioned and unversioned)
        info = tx_structure.get(tid) or tx_structure.get(safe_clean_id(tid))
        if not info or 'exons' not in info:
            n_skipped += 1
            continue

        pos_map = pos_maps.get(tid) or pos_maps.get(safe_clean_id(tid))
        if not pos_map:
            n_skipped += 1
            continue

        # Convert transcript-relative ORF coordinates to genomic
        strand = info['strand']
        chrom = info['chr']
        gene_id = info.get('gene_id', 'TRACE_gene')
        gene_name = info.get('gene_name', 'TRACE_gene')

        # Map start and stop positions
        gp_start = pos_map.get(orf_start)
        gp_stop = pos_map.get(orf_stop)
        if gp_start is None or gp_stop is None:
            n_skipped += 1
            continue

        genomic_start = min(gp_start, gp_stop)
        genomic_end = max(gp_start, gp_stop)

        # Deduplicate
        key = (tid, genomic_start, genomic_end)
        if key in seen: continue
        seen.add(key)

        attr = (f'gene_id "{gene_id}"; transcript_id "{tid}"; '
                f'gene_name "{gene_name}";')
        cds_line = (f'{chrom}\t{args.source_tag}\tCDS\t'
                    f'{genomic_start}\t{genomic_end}\t.\t{strand}\t0\t{attr}\n')
        trace_cds_lines.append(cds_line)
        trace_tx_set.add(tid)
        trace_tx_set.add(safe_clean_id(tid))

    print(f" -> {len(trace_cds_lines)} TRACE CDS lines for {len(trace_tx_set)} transcripts")
    if n_skipped:
        print(f" -> {n_skipped} ORFs skipped (no exon structure or out of bounds)")

    # ------------------------------------------------------------------
    # 4. Extract reference canonical CDS lines
    # ------------------------------------------------------------------
    print("\n--- Extracting reference canonical CDS ---")
    tx_re = re.compile(r'transcript_id "([^"]+)"')
    ref_cds_lines = []
    tx_has_ref_cds = set()

    with open(args.ref_gtf) as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split('\t')
            if len(parts) < 9 or parts[2] != 'CDS': continue
            tx_match = tx_re.search(parts[8])
            if tx_match:
                tid = tx_match.group(1).strip()
                ref_cds_lines.append(line)
                tx_has_ref_cds.add(tid)

    print(f" -> {len(tx_has_ref_cds)} transcripts with canonical CDS")

    # ------------------------------------------------------------------
    # 5. Assemble final GTF: TRACE CDS (base) + ref CDS for uncovered tx
    # ------------------------------------------------------------------
    print("\n--- Writing enhanced GTF ---")
    n_ref_appended = 0

    os.makedirs(os.path.dirname(args.output_gtf) or '.', exist_ok=True)
    with open(args.output_gtf, 'w') as fout:
        fout.write(f"# Enhanced GTF: TRACE CDS (genomic coords) + ref canonical CDS\n")
        fout.write(f"# TRACE ORFs: {len(trace_cds_lines)} CDS lines\n")

        for line in trace_cds_lines:
            fout.write(line)

        for line in ref_cds_lines:
            tx_match = tx_re.search(line)
            if tx_match:
                tid = tx_match.group(1).strip()
                if tid in trace_tx_set or safe_clean_id(tid) in trace_tx_set:
                    continue
            fout.write(line)
            n_ref_appended += 1

    print(f" -> Appended {n_ref_appended} reference CDS lines")
    print(f" -> Enhanced GTF saved: {args.output_gtf}")
    print(f" -> Coverage: TRACE={len(trace_tx_set)} + ref={n_ref_appended}")

if __name__ == "__main__":
    main()
