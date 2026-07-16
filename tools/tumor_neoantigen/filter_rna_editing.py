#!/usr/bin/env python3
"""
Filter RNA editing sites from a somatic VCF.

RNA editing (predominantly A-to-I and C-to-U) can generate apparent somatic
variants in RNA-seq data.  This script removes PASS variants that fall within
known RNA editing sites, using either:

  1. A BED file of known editing positions (e.g., from REDIportal / RADAR),
     specified via --editing_db.

  2. When --editing_db is not provided, an optional heuristic mode
     (--filter_a_to_g) removes A>G and T>C variants genome-wide as these
     are the most common RNA editing signatures (use with caution).

Output is a filtered VCF with the same header as the input.
"""
import os, sys, gzip, argparse

def main():
    p = argparse.ArgumentParser(
        description="Filter RNA editing sites from a somatic VCF.")
    p.add_argument("--vcf", required=True, help="Input VCF (.vcf or .vcf.gz)")
    p.add_argument("--output", required=True, help="Output filtered VCF")
    p.add_argument("--editing_db", default=None,
                   help="BED file of known editing sites (chrom, start, end)")
    p.add_argument("--filter_a_to_g", action="store_true",
                   help="Heuristic: remove all A>G and T>C variants "
                        "(common RNA editing signature).  Use only if no "
                        "editing database is available.")
    args = p.parse_args()

    # ------------------------------------------------------------------
    # 1. Load known editing sites (if provided)
    # ------------------------------------------------------------------
    editing_sites = set()  # (chrom, pos)
    if args.editing_db and os.path.exists(args.editing_db):
        print(f"--- Loading editing sites from: {args.editing_db}")
        with open(args.editing_db) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) < 3:
                    continue
                chrom = parts[0]
                start = int(parts[1])
                end = int(parts[2])
                # BED: start is 0-based, end is 1-based exclusive.
                # Each position in [start, end) is an editing site.
                for pos in range(start + 1, end + 1):  # 1-based positions
                    editing_sites.add((chrom, pos))
        print(f" -> Loaded {len(editing_sites)} editing positions")

    # ------------------------------------------------------------------
    # 2. Process VCF
    # ------------------------------------------------------------------
    vcf_open = gzip.open if args.vcf.endswith('.gz') else open
    n_total = 0
    n_filtered = 0

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with vcf_open(args.vcf, 'rt') as fin, open(args.output, 'w') as fout:
        for line in fin:
            # Pass through header lines unchanged
            if line.startswith('#'):
                fout.write(line)
                continue

            parts = line.strip().split('\t')
            if len(parts) < 8:
                fout.write(line)
                continue

            chrom = parts[0]
            pos = int(parts[1])
            ref = parts[3]
            alt = parts[4]
            filt = parts[6]

            # Only filter PASS variants
            if filt != 'PASS':
                fout.write(line)
                continue

            n_total += 1

            # Check against editing database
            if (chrom, pos) in editing_sites:
                n_filtered += 1
                continue  # skip this variant

            # Heuristic A>G / T>C filter
            if args.filter_a_to_g:
                alt_allele = alt.split(',')[0]
                if (ref.upper() == 'A' and alt_allele.upper() == 'G') or \
                   (ref.upper() == 'T' and alt_allele.upper() == 'C'):
                    n_filtered += 1
                    continue

            fout.write(line)

    pct = n_filtered / max(n_total, 1) * 100
    print(f" -> Total PASS variants: {n_total}")
    print(f" -> Filtered (RNA editing): {n_filtered} ({pct:.1f}%)")
    print(f" -> Remaining: {n_total - n_filtered}")
    print(f" -> Output: {args.output}")


if __name__ == "__main__":
    main()
