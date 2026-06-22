#!/usr/bin/env python3
import os
import re
import sys
import argparse

def clean_id(tid):
    tid_str = str(tid).strip()
    if tid_str.startswith('ENS'):
        return tid_str.split('.')[0]
    return tid_str


def parse_gtf_junctions(gtf_file):
    """
    Parse a GTF file to extract all exon-exon junctions for each transcript.
    Junctions are defined by the coordinates between adjacent exons.
    """
    print(f" -> Parsing GTF: {os.path.basename(gtf_file)}")
    tx_exons = {}
    
    # Regular expressions for speed
    tx_re = re.compile(r'transcript_id "([^"]+)"')
    
    with open(gtf_file, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.split('\t')
            if len(parts) < 9 or parts[2] != 'exon': continue
                
            info = parts[8]
            tx_match = tx_re.search(info)
            if not tx_match: continue
                
            # Strip Ensembl version suffixes for robust mapping
            tx_id = clean_id(tx_match.group(1))
            chrom = parts[0]
            start = int(parts[3])
            end = int(parts[4])
            strand = parts[6]
            
            tx_exons.setdefault(tx_id, []).append((chrom, start, end, strand))
            
    junction_map = {} 
    tx_to_juncs = {}  
    
    for tx_id, exons in tx_exons.items():
        if len(exons) < 2: continue
            
        chrom = exons[0][0]
        strand = exons[0][3]
        exons.sort(key=lambda x: x[1])
        
        tx_junc_set = set()
        for i in range(len(exons) - 1):
            # featureCounts Site1/Site2 logic: 
            # End of upstream exon & Start of downstream exon
            junc_start = exons[i][2]
            junc_end = exons[i+1][1]
            junc_id = f"{chrom}:{junc_start}-{junc_end}"
            
            tx_junc_set.add(junc_id)
            junction_map.setdefault(junc_id, set()).add(tx_id)
            
        tx_to_juncs[tx_id] = tx_junc_set
        
    return junction_map, tx_to_juncs

def main():
    parser = argparse.ArgumentParser(description="Extract theoretical transcript-specific junctions from GTF structures.")
    parser.add_argument("-r", "--ref_gtf", required=True, help="Baseline reference GTF file (e.g., GENCODE/Ensembl).")
    parser.add_argument("-t", "--tumor_gtfs", required=True, nargs='+', help="One or more tumor/assembled GTF files (e.g., StringTie output).")
    parser.add_argument("-o", "--output_mapping", required=True, help="Output TSV mapping file (Junction_ID -> Transcript_ID).")
    parser.add_argument("--mode", choices=['strict_unique', 'tumor_specific'], default='tumor_specific',
                        help="strict_unique: Maps to EXACTLY ONE transcript globally. tumor_specific: Novel junction OR unique reference junction.")
    
    args = parser.parse_args()

    print("--- Step 1: Processing Reference Baseline GTF ---")
    ref_junc_map, ref_tx_to_juncs = parse_gtf_junctions(args.ref_gtf)
    print(f" -> Found {len(ref_junc_map)} reference junctions across {len(ref_tx_to_juncs)} reference transcripts.")

    print("\n--- Step 2: Processing Tumor/Assembled GTFs ---")
    combined_junc_map = {}
    
    # Seed combined map with reference to track overall uniqueness
    for j_id, tx_set in ref_junc_map.items():
        combined_junc_map[j_id] = set(tx_set)
        
    for t_gtf in args.tumor_gtfs:
        t_junc_map, _ = parse_gtf_junctions(t_gtf)
        for j_id, tx_set in t_junc_map.items():
            combined_junc_map.setdefault(j_id, set()).update(tx_set)
            
    print(f" -> Consolidated topological architecture contains {len(combined_junc_map)} total unique junctions.")

    print("\n--- Step 3: Extracting Specific Targets based on Mode ---")
    valid_mappings = []
    
    for j_id, tx_set in combined_junc_map.items():
        if args.mode == 'strict_unique':
            if len(tx_set) == 1:
                valid_mappings.append(f"{j_id}\t{list(tx_set)[0]}\n")
        elif args.mode == 'tumor_specific':
            # =========================================================================
            # [CRITICAL UPGRADE]: Adaptive AND-Gate to preserve ENST specific junctions
            # =========================================================================
            # Keep the junction if:
            # 1. It is completely novel (not present in the normal reference baseline)
            # 2. OR it exists in reference but belongs strictly to ONE unique transcript (e.g., ENST)
            if (j_id not in ref_junc_map) or (len(tx_set) == 1):
                valid_mappings.append(f"{j_id}\t{','.join(list(tx_set))}\n")

    print(f" -> Identified {len(valid_mappings)} valid topological mappings ({args.mode}).")
    
    os.makedirs(os.path.dirname(args.output_mapping) or '.', exist_ok=True)
    with open(args.output_mapping, 'w') as out_f:
        out_f.write("Junction_ID\tTranscript_ID\n")
        out_f.writelines(valid_mappings)
        
    print(f"✅ Success! Saved theoretical junction dictionary to: {args.output_mapping}")

if __name__ == "__main__":
    main()