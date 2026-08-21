"""Prepare and annotate local RBP motif databases."""

def pre_annotate_and_save_database(combined_pwms, combined_meta, out_dir):
    """
    Pre-annotate merged metadata using the MyGene.info API and solidify the database locally.
    [Updated]: Dynamically extracts GO Biological Process (BP) terms alongside standard summaries
               to facilitate downstream functional clustering of RNA-binding proteins.
    """
    import os
    import requests
    import time
    import pickle
    import pandas as pd
    from tqdm import tqdm

    os.makedirs(out_dir, exist_ok=True)
    print("\n--- Phase 1: Pre-annotating Metadata with MyGene.info ---")
    
    unique_ensgs = combined_meta['Gene_id'].dropna().unique()
    print(f"Found {len(unique_ensgs)} unique Ensembl IDs to annotate.")
    
    # Cache summaries and GO biological-process annotations separately.
    summary_cache = {}
    go_bp_cache = {}
    
    for ensg in tqdm(unique_ensgs, desc="Fetching API"):
        ensg_clean = str(ensg).strip()
        if not ensg_clean.startswith("ENSG"):
            summary_cache[ensg] = "Unannotated (Invalid ID)"
            go_bp_cache[ensg] = "None"
            continue
            
        # Request GO biological-process terms together with gene summaries.
        url = f"https://mygene.info/v3/gene/{ensg_clean}?fields=summary,name,go.BP"
        
        try:
            time.sleep(0.1)  # Rate limiting safety buffer
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract the primary functional summary.
                func_desc = data.get('summary', data.get('name', 'Summary unavailable in NCBI.'))
                summary_cache[ensg] = func_desc
                
                # Extract GO biological-process annotations.
                go_data = data.get('go', {})
                bp_entries = go_data.get('BP', [])
                
                # MyGene returns one term as a dict and multiple terms as a list.
                bp_terms = []
                if isinstance(bp_entries, list):
                    for entry in bp_entries:
                        term = entry.get('term')
                        if term: bp_terms.append(term)
                elif isinstance(bp_entries, dict):
                    term = bp_entries.get('term')
                    if term: bp_terms.append(term)
                
                # Join unique biological-process terms into one field.
                if bp_terms:
                    go_bp_cache[ensg] = "; ".join(sorted(list(set(bp_terms))))
                else:
                    go_bp_cache[ensg] = "No BP terms annotated"
                    
            elif response.status_code == 404:
                summary_cache[ensg] = "Gene not found in MyGene."
                go_bp_cache[ensg] = "None"
            else:
                summary_cache[ensg] = f"HTTP {response.status_code}"
                go_bp_cache[ensg] = "None"
                
        except Exception:
            summary_cache[ensg] = "API Fetch Error"
            go_bp_cache[ensg] = "None"

    # Map both annotation groups back to the combined metadata table.
    combined_meta['RBP_Function'] = combined_meta['Gene_id'].map(summary_cache)
    combined_meta['RBP_GO_BP'] = combined_meta['Gene_id'].map(go_bp_cache)
    
    print("\n--- Phase 2: Saving Unified Database to Disk ---")
    
    # Save complete metadata with functional and GO annotations.
    meta_save_path = os.path.join(out_dir, "Unified_RBP_Metadata_Annotated.tsv")
    combined_meta.to_csv(meta_save_path, sep='\t', index=False)
    print(f"✅ Annotated Metadata saved: {meta_save_path}")
    
    # Preserve the NumPy PWM dictionary without lossy conversion.
    pwm_save_path = os.path.join(out_dir, "Unified_RBP_PWMs.pkl")
    with open(pwm_save_path, 'wb') as f:
        pickle.dump(combined_pwms, f)
    print(f"✅ Unified PWM Dictionary saved: {pwm_save_path}")
    
    return combined_meta
