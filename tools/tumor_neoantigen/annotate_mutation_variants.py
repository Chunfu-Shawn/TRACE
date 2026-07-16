#!/usr/bin/env python3
"""
Annotate somatic variants from VCF with transcript-level coding impact.

Parses a GTF to build a CDS position map for each transcript, then annotates
every PASS variant in a VCF with:
  - Affected transcript(s) and gene(s)
  - Codon and amino acid change
  - Mutation classification (synonymous / nonsynonymous / stop_gain / stop_loss)

Output is a CSV of nonsynonymous coding variants for downstream neoantigen prediction.
"""
import os, re, sys, gzip, argparse
import pandas as pd
from collections import defaultdict

CODON_TABLE = {
    'ATA':'I','ATC':'I','ATT':'I','ATG':'M','ACA':'T','ACC':'T','ACG':'T','ACT':'T',
    'AAC':'N','AAT':'N','AAA':'K','AAG':'K','AGC':'S','AGT':'S','AGA':'R','AGG':'R',
    'CTA':'L','CTC':'L','CTG':'L','CTT':'L','TTA':'L','TTG':'L',
    'CCA':'P','CCC':'P','CCG':'P','CCT':'P','CAC':'H','CAT':'H','CAA':'Q','CAG':'Q',
    'CGA':'R','CGC':'R','CGG':'R','CGT':'R','GTA':'V','GTC':'V','GTG':'V','GTT':'V',
    'GCA':'A','GCC':'A','GCG':'A','GCT':'A','GAC':'D','GAT':'D','GAA':'E','GAG':'E',
    'GGA':'G','GGC':'G','GGG':'G','GGT':'G','TCA':'S','TCC':'S','TCG':'S','TCT':'S',
    'TTC':'F','TTT':'F','TAC':'Y','TAT':'Y','TGC':'C','TGT':'C','TGG':'W',
    'TAA':'*','TAG':'*','TGA':'*',
}
COMPLEMENT = str.maketrans('ATCGatcg', 'TAGCtagc')

def revcomp(s): return s.translate(COMPLEMENT)[::-1]
def translate(codon): return CODON_TABLE.get(codon.upper(), 'X')

def build_cds_map(gtf_path):
    """Build {transcript_id: {chr, strand, gene_name, pos_map: {genomic_pos: coding_pos}}}"""
    tx_cds = defaultdict(list)
    tx_info = {}
    tx_re = re.compile(r'transcript_id "([^"]+)"')
    gene_re = re.compile(r'gene_id "([^"]+)"')
    gname_re = re.compile(r'gene_name "([^"]+)"')
    opener = gzip.open if gtf_path.endswith('.gz') else open
    with opener(gtf_path, 'rt') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split('\t')
            if len(parts) < 9 or parts[2] != 'CDS': continue
            info = parts[8]
            tx_match = tx_re.search(info)
            if not tx_match: continue
            tx_id = tx_match.group(1).strip()
            tx_cds[tx_id].append((parts[0], int(parts[3]), int(parts[4]), parts[6], int(parts[7])))
            if tx_id not in tx_info:
                gm = gene_re.search(info)
                gn = gname_re.search(info)
                tx_info[tx_id] = {'chr': parts[0], 'strand': parts[6],
                                  'gene_id': gm.group(1).strip() if gm else 'Unknown',
                                  'gene_name': gn.group(1).strip() if gn else 'Unknown'}
    cds_map = {}
    for tx_id, exons in tx_cds.items():
        exons.sort(key=lambda x: x[1])
        info = tx_info[tx_id]; strand = info['strand']
        pos_map = {}; coding_pos = 1
        for _, start, end, _, phase in exons:
            exon_len = end - start + 1
            for off in range(exon_len):
                gp = start + off
                pos_map[gp] = coding_pos + (off if strand == '+' else exon_len - 1 - off)
            coding_pos += exon_len
        cds_map[tx_id] = {**info, 'cds_length': coding_pos - 1, 'pos_map': pos_map}
    print(f"Built CDS map: {len(cds_map)} transcripts")
    return cds_map

def annotate_vcf(vcf_path, cds_map):
    """Annotate PASS variants against CDS map. Returns list of dicts."""
    results = []
    n_total = 0; n_coding = 0; n_nonsyn = 0
    vcf_open = gzip.open if vcf_path.endswith('.gz') else open
    with vcf_open(vcf_path, 'rt') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split('\t')
            if len(parts) < 8: continue
            chrom, pos, _, ref, alt, _, filt = parts[0], int(parts[1]), parts[2], parts[3], parts[4], parts[5], parts[6]
            if filt != 'PASS': continue
            n_total += 1
            alt_allele = alt.split(',')[0]
            for tx_id, tx_data in cds_map.items():
                if tx_data['chr'] != chrom: continue
                pos_map = tx_data.get('pos_map', {})
                if pos not in pos_map: continue
                n_coding += 1
                strand = tx_data['strand']
                cp = pos_map[pos]
                codon_idx = (cp - 1) // 3
                codon_off = (cp - 1) % 3
                # Build codon: find the 3 genomic positions
                codon_gpos = [None, None, None]
                for gp, c in pos_map.items():
                    c0 = c - 1
                    if c0 // 3 == codon_idx:
                        codon_gpos[c0 % 3] = gp
                if not all(g is not None for g in codon_gpos): continue
                ref_bases = ['N','N','N']; alt_bases = ['N','N','N']
                for j, gp in enumerate(codon_gpos):
                    if gp == pos:
                        ref_bases[j] = ref; alt_bases[j] = alt_allele
                ref_codon = ''.join(ref_bases); alt_codon = ''.join(alt_bases)
                if strand == '-':
                    ref_codon = revcomp(ref_codon); alt_codon = revcomp(alt_codon)
                ref_aa = translate(ref_codon); alt_aa = translate(alt_codon)
                if ref_aa == alt_aa: mut_type = 'synonymous'
                elif alt_aa == '*': mut_type = 'stop_gain'
                elif ref_aa == '*': mut_type = 'stop_loss'
                else: mut_type = 'nonsynonymous'; n_nonsyn += 1
                aa_change = f"{ref_aa}{codon_idx+1}{alt_aa}" if ref_aa != alt_aa else ''
                results.append({'Chrom':chrom,'Pos':pos,'Ref':ref,'Alt':alt_allele,
                    'Transcript_ID':tx_id,'Gene_Name':tx_data['gene_name'],
                    'Gene_ID':tx_data['gene_id'],'Strand':strand,
                    'Codon_Pos':codon_idx+1,'Coding_Pos':cp,
                    'Ref_Codon':ref_codon,'Alt_Codon':alt_codon,
                    'Ref_AA':ref_aa,'Alt_AA':alt_aa,'AA_Change':aa_change,
                    'Mutation_Type':mut_type})
    print(f"PASS variants: {n_total}, coding: {n_coding}, nonsyn: {n_nonsyn}")
    return results

def main():
    p = argparse.ArgumentParser(description="Annotate somatic VCF with coding impact")
    p.add_argument("--vcf", required=True, help="Input VCF (.vcf or .vcf.gz)")
    p.add_argument("--gtf", required=True, help="Reference GTF annotation")
    p.add_argument("--output", required=True, help="Output CSV")
    p.add_argument("--rna_editing_db", default=None, help="Optional BED of known editing sites")
    args = p.parse_args()
    print("--- Building CDS map ---")
    cds_map = build_cds_map(args.gtf)
    print("\n--- Annotating variants ---")
    results = annotate_vcf(args.vcf, cds_map)
    if not results:
        print("[Warning] No coding variants found.")
        pd.DataFrame().to_csv(args.output, index=False)
        return
    df = pd.DataFrame(results)
    if args.rna_editing_db and os.path.exists(args.rna_editing_db):
        print(f"\n--- Filtering RNA editing sites ---")
        edit_df = pd.read_csv(args.rna_editing_db, sep='\t', header=None,
                              names=['Chrom','Start','End','Type'])
        n_before = len(df)
        for _, row in edit_df.iterrows():
            df = df[~((df['Chrom']==row['Chrom'])&(df['Pos']>=row['Start'])&(df['Pos']<=row['End']))]
        print(f"Removed {n_before - len(df)} potential RNA editing sites")
    df_nonsyn = df[df['Mutation_Type']=='nonsynonymous'].copy()
    print(f"Nonsynonymous: {len(df_nonsyn)}")
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    df_nonsyn.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")

if __name__=="__main__": main()
