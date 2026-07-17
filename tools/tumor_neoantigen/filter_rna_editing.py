#!/usr/bin/env python3
"""
Filter RNA editing sites from a somatic VCF using REDIportal or BED databases.

Supports two database formats:
  1. REDIportal TSV (columns: Accession, Region, Position, Ref, Ed, Strand,
     db, type, dbsnp, repeat, ...)
  2. BED format (chrom, start, end)

When a database is provided, PASS variants whose (chrom, pos) match a known
editing site are removed.  Without a database, an optional heuristic mode
(--filter_a_to_g) removes A>G and T>C variants genome-wide.

Usage:
    python filter_rna_editing.py \
        --vcf input.vcf.gz \
        --editing_db REDIportal_ATLAS_2024.tsv \
        --output filtered.vcf
"""
import os, sys, gzip, argparse, pandas as pd

def load_editing_sites(db_path):
    """
    Load known editing positions from a database file.

    Auto-detects format:
      - .tsv/.txt: REDIportal format (columns: Region, Position, ...)
      - .bed: BED format (chrom, start, end)
    Returns set of (chrom, pos) tuples.
    """
    sites = set()
    ext = os.path.splitext(db_path)[1].lower()

    if ext in ('.tsv', '.txt', '.csv'):
        print(f" -> Detected TSV format (REDIportal)")
        df = pd.read_csv(db_path, sep='\t', low_memory=False)
        # Standardize column names (case-insensitive)
        cols_lower = {c.lower(): c for c in df.columns}
        region_col = cols_lower.get('region', 'Region')
        pos_col = cols_lower.get('position', 'Position')
        if region_col not in df.columns or pos_col not in df.columns:
            print(f"[Error] TSV missing 'Region' or 'Position' column. Found: {list(df.columns)[:10]}")
            sys.exit(1)
        for _, row in df.iterrows():
            chrom = str(row[region_col]).strip()
            pos = int(row[pos_col])
            sites.add((chrom, pos))
    elif ext == '.bed':
        print(f" -> Detected BED format")
        with open(db_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('track'):
                    continue
                parts = line.split('\t')
                if len(parts) < 3:
                    continue
                chrom = parts[0]
                start = int(parts[1])
                end = int(parts[2])
                for pos in range(start + 1, end + 1):
                    sites.add((chrom, pos))
    else:
        print(f"[Error] Unrecognized database format: {ext}. Expected .tsv or .bed")
        sys.exit(1)

    return sites

def main():
    p = argparse.ArgumentParser(
        description="Filter RNA editing sites from a somatic VCF.")
    p.add_argument("--vcf", required=True, help="Input VCF (.vcf or .vcf.gz)")
    p.add_argument("--output", required=True, help="Output filtered VCF")
    p.add_argument("--editing_db", default=None,
                   help="Path to RNA editing database (REDIportal TSV or BED)")
    p.add_argument("--filter_a_to_g", action="store_true",
                   help="Heuristic: remove all A>G and T>C variants "
                        "(common RNA editing signatures). Use only if no "
                        "editing database is available.")
    args = p.parse_args()

    # ------------------------------------------------------------------
    # 1. Load editing sites
    # ------------------------------------------------------------------
    editing_sites = set()
    if args.editing_db and os.path.exists(args.editing_db):
        print(f"--- Loading editing sites from: {args.editing_db}")
        editing_sites = load_editing_sites(args.editing_db)
        print(f" -> Loaded {len(editing_sites):,} unique editing positions")
    elif args.editing_db:
        print(f"[Warning] Editing DB not found: {args.editing_db}")
        if args.filter_a_to_g:
            print(" -> Falling back to heuristic A>G / T>C filter")
        else:
            print(" -> No filter will be applied (pass --filter_a_to_g for heuristic)")
    elif args.filter_a_to_g:
        print("--- Heuristic A>G / T>C filter enabled (no database) ---")
    else:
        print("--- No editing filter (no database, no heuristic) ---")

    # ------------------------------------------------------------------
    # 2. Process VCF
    # ------------------------------------------------------------------
    vcf_open = gzip.open if args.vcf.endswith('.gz') else open
    n_total = 0
    n_filtered_db = 0
    n_filtered_heuristic = 0

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with vcf_open(args.vcf, 'rt') as fin, open(args.output, 'w') as fout:
        for line in fin:
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

            if filt != 'PASS':
                fout.write(line)
                continue

            n_total += 1

            # Database filter
            if editing_sites and (chrom, pos) in editing_sites:
                n_filtered_db += 1
                continue

            # Heuristic A>G / T>C filter
            if args.filter_a_to_g:
                alt_allele = alt.split(',')[0]
                if (ref.upper() == 'A' and alt_allele.upper() == 'G') or \
                   (ref.upper() == 'T' and alt_allele.upper() == 'C'):
                    n_filtered_heuristic += 1
                    continue

            fout.write(line)

    n_filtered = n_filtered_db + n_filtered_heuristic
    pct = n_filtered / max(n_total, 1) * 100
    print(f"\n--- Filtering summary ---")
    print(f"  Total PASS variants:        {n_total}")
    if n_filtered_db:
        print(f"  Removed (editing database):  {n_filtered_db}")
    if n_filtered_heuristic:
        print(f"  Removed (A>G / T>C heuristic): {n_filtered_heuristic}")
    print(f"  Remaining:                   {n_total - n_filtered} ({100 - pct:.1f}%)")
    print(f"  Output: {args.output}")

if __name__ == "__main__":
    main()
