import os
import pickle
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
from plotnine import *
import warnings
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from eval.calculate_te import *

# Ignore warnings
warnings.filterwarnings("ignore")

def extract_gt_and_cds_from_datasets(datasets):
    """
    Extract Ground Truth (GT) sequences and CDS information from single or multiple TranslationDatasets.
    Returns: { cell_type: { tid: {'gt': [...], 'cds_start': x, 'cds_end': y, 'depth': z} } }
    """
    print("Extracting Ground Truth and CDS info from datasets...")
    info_dict = defaultdict(dict)
    
    # Ensure datasets is a list for uniform processing
    if not isinstance(datasets, (list, tuple)):
        datasets = [datasets]
        
    for d_idx, dataset in enumerate(datasets):
        print(f"Parsing Dataset {d_idx + 1}/{len(datasets)}...")
        for i in tqdm(range(len(dataset)), desc=f"Dataset {d_idx + 1}"):
            uuid, species, cell_type, expr_vector, meta_info, seq_emb, count_emb = dataset[i]
            
            # Use the cell_type directly from dataset, NOT from uuid parsing
            # (uuid-based parsing may not match the actual metadata cell_type)
            parts = str(uuid).rsplit('-', 2)
            if len(parts) < 2: 
                continue
            tid = parts[0]
            # cell_type already comes from dataset[i]; do not overwrite
            
            if torch.is_tensor(count_emb):
                gt_array = count_emb.numpy().reshape(-1)
            else:
                gt_array = np.array(count_emb).reshape(-1)
                
            # [Fix]: Revert log1p transformation back to linear scale
            gt_array = np.expm1(gt_array)
            
            info_dict[cell_type][tid] = {
                'gt': gt_array,
                'cds_start': meta_info['cds_start_pos'],
                'cds_end': meta_info['cds_end_pos'],
                'rpf_depth': meta_info.get('rpf_depth', 0) 
            }
            
    print(f"Extraction complete for {len(info_dict)} unique cell types across all datasets.")
    return info_dict

class MultiCellEvaluator:
    def __init__(self, datasets, pkl_paths, min_depth=1.0, min_cells=3):
        """
        Initialize with single or multiple datasets and pkl files.
        """
        # Accept either single path/dataset or a list of them
        self.datasets = datasets if isinstance(datasets, (list, tuple)) else [datasets]
        self.pkl_paths = pkl_paths if isinstance(pkl_paths, (list, tuple)) else [pkl_paths]
        
        self.min_depth = min_depth
        self.min_cells = min_cells
        
        # Pass the list of datasets to the extraction function
        self.dataset_info = extract_gt_and_cds_from_datasets(self.datasets)
        
        self.grouped_data = self._load_and_group_data()
        self.cell_types = self._get_all_cell_types()
        
        self.transcript_metrics_df = None 
        self.te_pairwise_data = None
        self._analysis_done = False

    def _load_and_group_data(self):
        """
        Iterate through multiple pkl files and merge all predictions into a single grouped dictionary.
        """
        grouped = defaultdict(dict)
        
        for pkl_path in self.pkl_paths:
            print(f"Loading predictions from {pkl_path}...")
            with open(pkl_path, 'rb') as f:
                raw_data = pickle.load(f)
            
            for cell_type, tid_dict in raw_data.items():
                for tid, pred_array in tid_dict.items():
                    try:
                        # Check if the (cell_type, tid) pair exists in the combined dataset info
                        if cell_type not in self.dataset_info or tid not in self.dataset_info[cell_type]:
                            continue
                            
                        ds_info = self.dataset_info[cell_type][tid]
                        
                        cds_start = ds_info['cds_start']
                        cds_end = ds_info['cds_end']
                        gt_array = ds_info['gt']
                        depth = ds_info.get('rpf_depth', 0)
                        
                        if cds_start < 0 or cds_end < 0:
                            continue
                        
                        # Merge data seamlessly; existing tid will just append new cell_types
                        grouped[tid][cell_type] = {
                            # [Fix]: Revert log1p transformation for predictions
                            'pred': np.expm1(np.array(pred_array).reshape(-1)),
                            'gt': gt_array, 
                            'cds_start': int(cds_start) - 1,  # 1-based -> 0-based
                            'cds_end': int(cds_end) - 1,  # 1-based -> 0-based (Python slice end is exclusive)
                            'depth': float(depth)
                        }
                    except Exception as e:
                        continue
                        
        print(f"Successfully integrated {len(self.pkl_paths)} files. Grouped into {len(grouped)} unique transcripts.")
        return grouped

    def _get_all_cell_types(self):
        cells = set()
        for t_data in self.grouped_data.values():
            cells.update(t_data.keys())
        return sorted(list(cells))

    def _run_global_analysis(self):
        """
        Calculates Transcript TE correlation (Specificity Score) across different cell lines.
        """
        if self._analysis_done: return

        print("Running global TE analysis...")
        transcript_results = []
        
        for tid, cells_dict in tqdm(self.grouped_data.items(), desc="Analyzing transcripts"):
            # Filter and prepare data
            available_cells = sorted(list(cells_dict.keys()))
            if len(available_cells) < self.min_cells: continue
            
            avg_depth = np.mean([cells_dict[c]['depth'] for c in available_cells])
            if avg_depth < self.min_depth: continue
            
            valid_cells = []
            gt_te_list = []
            pred_te_list = []
            
            for c in available_cells:
                g = cells_dict[c]['gt']
                p = cells_dict[c]['pred']
                cds_start = cells_dict[c]['cds_start']
                cds_end = cells_dict[c]['cds_end']
                
                # Filter out arrays with zero variance
                if np.std(g) > 1e-6 or np.std(p) > 1e-6:
                    valid_cells.append(c)
                    
                    # Calculate TE using morf mean signal of frame0 in CDS
                    gt_te = calculate_morf_mean_signal(g, cds_start, cds_end)
                    pred_te = calculate_morf_mean_signal(p, cds_start, cds_end)
                    
                    gt_te_list.append(gt_te)
                    pred_te_list.append(pred_te)
            
            # Require at least 3 valid cells to compute a meaningful spearman correlation
            if len(valid_cells) < 3: continue 
            
            gt_te_arr = np.array(gt_te_list, dtype=np.float32)
            pred_te_arr = np.array(pred_te_list, dtype=np.float32)
            
            # Calculate spearman correlation between GT TE and Pred TE across cells
            if np.std(gt_te_arr) > 1e-6 and np.std(pred_te_arr) > 1e-6:
                te_corr, _ = spearmanr(gt_te_arr, pred_te_arr)
            else:
                te_corr = np.nan

            transcript_results.append({
                'TID': tid,
                'N_Cells': len(valid_cells),
                'Avg_Depth': avg_depth,
                'Specificity_Score': te_corr 
            })
            
        self.transcript_metrics_df = pd.DataFrame(transcript_results).dropna()
        self._analysis_done = True
        print("Global analysis finished.")

    def evaluate_specificity(self, out_dir="./results"):
        """
        Saves and returns the TE specificty results for each transcript.
        """
        self._run_global_analysis()
        
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "te_specificity_results.csv")
        self.transcript_metrics_df.to_csv(csv_path, index=False)
        print(f"TE Specificity results saved to {csv_path}")
        return self.transcript_metrics_df

    def plot_specificity_vs_depth(self, out_path="specificity_vs_depth_scatter.pdf"):
        """
        Plots the correlation between TE Specificity Score and Sequencing Depth.
        """
        df = self.evaluate_specificity(out_dir=os.path.dirname(out_path) or ".")
        
        df = df.copy()
        df['Log_Depth'] = np.log1p(df['Avg_Depth'])
        clean_df = df.dropna(subset=['Log_Depth', 'Specificity_Score'])
        
        if len(clean_df) < 2: return

        r, p = spearmanr(clean_df['Log_Depth'], clean_df['Specificity_Score'])
        stats_label = (f"spearman R = {r:.3f}\n(P={p:.2e})")
        
        p_plot = (
            ggplot(clean_df, aes(x='Log_Depth', y='Specificity_Score'))
            + geom_point(alpha=0.3, color="#2d3436", size=2, stroke=0)
            + geom_smooth(method='lm', color="#005b96", size=1)
            + annotate("text", x=clean_df["Log_Depth"].min(), y=clean_df['Specificity_Score'].max(), 
                       label=stats_label, ha='left', va='top', size=10)
            + labs(x="log(average P-site depth + 1) in Obs.", 
                   y="TE Correlation across cells (Pred vs Obs)")
            + theme_classic()
            + theme(figure_size=(4, 4))
        )
        p_plot.save(out_path)
        print(f"Saved: {out_path}")

    def compute_te_pairwise_matrices(self, out_dir="./results", log_transform=True):
        """
        Extracts TE for all transcripts and computes the pairwise TE correlation matrix between cell types.
        """
        print("Calculating Transcript TE and pairwise correlations...")
        os.makedirs(out_dir, exist_ok=True)
        
        te_records = []
        for tid, cells_dict in self.grouped_data.items():
            for cell, data in cells_dict.items():
                cds_start, cds_end = data['cds_start'], data['cds_end']
                
                gt_te = calculate_morf_mean_signal(data['gt'], cds_start, cds_end)
                pred_te = calculate_morf_mean_signal(data['pred'], cds_start, cds_end)
                
                te_records.append({
                    'TID': tid,
                    'Cell': cell,
                    'GT_TE': gt_te,
                    'Pred_TE': pred_te
                })
        
        te_df = pd.DataFrame(te_records)
        
        # Force conversion to float32 to prevent Pandas Cython TypeError
        te_df['GT_TE'] = te_df['GT_TE'].astype('float32')
        te_df['Pred_TE'] = te_df['Pred_TE'].astype('float32')
        te_df.to_csv(os.path.join(out_dir, "transcript_TE_values.csv"), index=False)
        
        # Pivot tables (Rows: TID, Columns: Cell Types)
        gt_pivot = te_df.pivot_table(index='TID', columns='Cell', values='GT_TE', aggfunc='mean')
        pred_pivot = te_df.pivot_table(index='TID', columns='Cell', values='Pred_TE', aggfunc='mean')
        
        # Align GT and Pred pivots: keep only (tid, cell) pairs present in BOTH
        common_mask = gt_pivot.notna() & pred_pivot.notna()
        gt_pivot = gt_pivot.where(common_mask)
        pred_pivot = pred_pivot.where(common_mask)
        
        # Drop rows (transcripts) with no valid cells left after alignment
        valid_rows = common_mask.any(axis=1)
        gt_pivot = gt_pivot.loc[valid_rows]
        pred_pivot = pred_pivot.loc[valid_rows]
        print(f"Aligned pivot: {gt_pivot.shape[0]} transcripts x {gt_pivot.shape[1]} cells (shared mask)")
        
        # Log transformation to prevent highly expressed housekeeping genes from dominating spearman correlation
        if log_transform:
            gt_pivot = np.log1p(gt_pivot)
            pred_pivot = np.log1p(pred_pivot)
        
        # Store pivots for external use (e.g., intersection with TPM)
        self.gt_pivot = gt_pivot
        self.pred_pivot = pred_pivot
            
        gt_corr_matrix = gt_pivot.corr(method="spearman", min_periods=5)
        pred_corr_matrix = pred_pivot.corr(method="spearman", min_periods=5)
        
        cells = self.cell_types
        records = []
        self.te_pairwise_data = {} 
        
        for i, c1 in enumerate(cells):
            for j, c2 in enumerate(cells):
                if c1 in gt_corr_matrix.columns and c2 in gt_corr_matrix.columns:
                    val_gt = gt_corr_matrix.loc[c1, c2]
                    val_pred = pred_corr_matrix.loc[c1, c2]
                else:
                    val_gt, val_pred = np.nan, np.nan
                    
                self.te_pairwise_data[(c1, c2)] = {'GT': val_gt, 'Pred': val_pred}
                
                if i < j: 
                    records.append({
                        "Cell 1": c1, "Cell 2": c2,
                        "GT TE Corr": val_gt, "Pred TE Corr": val_pred
                    })
                    
        df_corr = pd.DataFrame(records).dropna()
        csv_path = os.path.join(out_dir, "cell_type_TE_pairwise_correlation.csv")
        df_corr.to_csv(csv_path, index=False)
        print(f"TE Pairwise correlation table saved to {csv_path}")
        
        return df_corr

    def extract_square_matrices_from_evaluator(self):
        """
        Reconstruct full square correlation DataFrames (GT and Pred) 
        from the te_pairwise_data dictionary in MultiCellEvaluator.
        """
        cells = self.cell_types
        
        # Initialize empty dataframes with 1.0 on the diagonal (self-correlation is always 1)
        gt_matrix = pd.DataFrame(1.0, index=cells, columns=cells)
        pred_matrix = pd.DataFrame(1.0, index=cells, columns=cells)
        
        # Check if data exists
        if getattr(self, 'te_pairwise_data', None) is None:
            raise ValueError("TE pairwise data not found. Please run self.compute_te_pairwise_matrices() first.")
        
        # Populate the matrices
        for (c1, c2), metrics in self.te_pairwise_data.items():
            # Ensure we don't overwrite diagonal 1.0 with NaN accidentally if it happened in computation
            if c1 == c2:
                continue
                
            gt_val = metrics.get('GT', np.nan)
            pred_val = metrics.get('Pred', np.nan)
            
            # Assign values symmetrically to build the full square matrix
            gt_matrix.loc[c1, c2] = gt_val
            gt_matrix.loc[c2, c1] = gt_val
            
            pred_matrix.loc[c1, c2] = pred_val
            pred_matrix.loc[c2, c1] = pred_val
            
        print(f"Successfully reconstructed {len(cells)}x{len(cells)} correlation matrices.")
        return pred_matrix, gt_matrix

    def plot_te_merged_heatmap(self, out_path="te_merged_heatmap.pdf"):
        """
        Plots the triangular heatmap for TE correlation between cells (GT upper, Pred lower).
        """
        if not hasattr(self, 'te_pairwise_data') or self.te_pairwise_data is None:
            self.compute_te_pairwise_matrices(out_dir=os.path.dirname(out_path) or ".", log_transform=True)
            
        cells = self.cell_types
        plot_data = []
        
        for i, c1 in enumerate(cells):
            for j, c2 in enumerate(cells):
                if i == j:
                    val = 1.0
                elif i < j: # Upper -> GT
                    val = self.te_pairwise_data.get((c1, c2), {}).get('GT', np.nan)
                else:       # Lower -> Pred
                    val = self.te_pairwise_data.get((c1, c2), {}).get('Pred', np.nan)
                    
                plot_data.append({'Cell_X': c2, 'Cell_Y': c1, 'Correlation': val})
                
        df_plot = pd.DataFrame(plot_data)
        df_plot['Cell_X'] = pd.Categorical(df_plot['Cell_X'], categories=cells)
        df_plot['Cell_Y'] = pd.Categorical(df_plot['Cell_Y'], categories=list(reversed(cells)))

        p_plot = (
            ggplot(df_plot, aes(x='Cell_X', y='Cell_Y', fill='Correlation'))
            + geom_tile(color="white", size=0.5)
            # + geom_text(aes(label='Correlation'), format_string='{:.2f}', size=8)
            + scale_fill_distiller(palette="OrRd", direction=-1, limits=(0, 1)) 
            + labs(title="Transcript TE Correlation: Obs. (Upper) vs Pred. (Lower)", x="", y="")
            + theme_minimal()
            + theme(axis_text_x=element_text(rotation=45, hjust=1), figure_size=(8, 7), panel_grid=element_blank())
        )
        p_plot.save(out_path)
        print(f"Saved TE Heatmap: {out_path}")

    def analyze_high_variance_te_transcripts(self, out_dir="./results", top_k=100, min_cells=3):
        """
        Identifies the Top K transcripts with the highest TE variance across cells and plots a clustermap.
        """
        print(f"Finding Top {top_k} transcripts with highest TE variance across cells...")
        os.makedirs(out_dir, exist_ok=True)
        
        te_records = []
        for tid, cells_dict in self.grouped_data.items():
            if len(cells_dict) < min_cells: continue
            
            for cell, data in cells_dict.items():
                te = calculate_morf_mean_signal(data['pred'], data['cds_start'], data['cds_end'])
                te_records.append({'TID': tid, 'Cell': cell, 'TE': te})
                
        if not te_records:
            print("No valid transcripts found based on min_cells threshold.")
            return
            
        te_df = pd.DataFrame(te_records)
        
        # [Fix]: Force conversion to float32 to prevent Pandas Cython TypeError caused by float16
        te_df['TE'] = te_df['TE'].astype('float32')
        
        # Pivot table, fill missing expression data with 0
        pivot_df = te_df.pivot_table(index='TID', columns='Cell', values='TE').fillna(0)
        
        # Calculate standard deviation as the variance metric
        pivot_df['TE_Std'] = pivot_df.std(axis=1)
        pivot_df['TE_Mean'] = pivot_df.mean(axis=1)
        pivot_df['TE_CV'] = pivot_df['TE_Std'] / (pivot_df['TE_Mean'] + 1e-6) 
        
        # Extract Top K (descending order by standard deviation)
        top_k_df = pivot_df.sort_values(by='TE_Std', ascending=False).head(top_k)
        
        csv_path = os.path.join(out_dir, f"top_{top_k}_variable_TE_transcripts.csv")
        top_k_df.reset_index().to_csv(csv_path, index=False)
        print(f"Saved Top {top_k} list to {csv_path}")
        
        # Plot bidirectional hierarchical clustermap
        heatmap_matrix = top_k_df.drop(columns=['TE_Std', 'TE_Mean', 'TE_CV'])
        
        # Log transformation for better visualization
        plot_matrix = np.log1p(heatmap_matrix)
        
        plt.figure(figsize=(10, 12))
        g = sns.clustermap(
            plot_matrix, 
            cmap="YlOrRd",           
            method='ward',           
            metric='euclidean',      
            figsize=(12, min(20, 4 + 0.15 * top_k)), 
            yticklabels=True, 
            xticklabels=True
        )
        g.fig.suptitle(f"Hierarchical Clustering of Top {top_k} Variable TEs (log1p)", y=1.02, fontsize=14)
        g.ax_heatmap.set_ylabel("Transcripts")
        g.ax_heatmap.set_xlabel("Cell Types")
        
        clustermap_path = os.path.join(out_dir, f"top_{top_k}_te_clustermap.pdf")
        g.savefig(clustermap_path, bbox_inches='tight')
        plt.close()
        print(f"Saved Clustermap to {clustermap_path}")


def load_and_calculate_tpm_correlation(tpm_file, mapping_file, target_cells=None, target_transcripts=None,
                                        te_pivot=None, log_transform=True):
    """
    Load TPM data, optionally filter by target transcripts (mapped to genes), 
    and compute the pairwise spearman correlation matrix between cell types.
    """
    print("Loading TPM and mapping data...")
    # Load TPM data
    tpm_df = pd.read_csv(tpm_file, index_col=0)
    
    # [Fix] Remove version numbers from TPM index (e.g., ENSG00000290825.1 -> ENSG00000290825)
    tpm_df.index = tpm_df.index.str.split('.').str[0]
    # Handle possible duplicate genes after version removal by taking the mean
    tpm_df = tpm_df.groupby(tpm_df.index).mean()
    
    # Optional: Filter cells to match only the ones we evaluated
    if target_cells is not None:
        valid_cells = [c for c in target_cells if c in tpm_df.columns]
        tpm_df = tpm_df[valid_cells]
        print(f"Filtered TPM data to {len(valid_cells)} target cells.")
    
    # Optional: Filter genes based on the specific transcripts we evaluated
    if target_transcripts is not None and mapping_file is not None:
        # Load mapping file
        mapping_df = pd.read_csv(mapping_file, sep='\t')
        
        # [Fix] Remove version numbers from the mapping file to ensure pure ID matching
        # Strip potential leading/trailing whitespaces in column names first
        mapping_df.columns = mapping_df.columns.str.strip()
        
        # Identify correct column names (handling slight variations)
        tx_col = [c for c in mapping_df.columns if 'Transcript' in c and 'ID' in c][0]
        gene_col = [c for c in mapping_df.columns if 'Gene' in c and 'ID' in c][0]
        
        mapping_df[tx_col] = mapping_df[tx_col].astype(str).str.split('.').str[0]
        mapping_df[gene_col] = mapping_df[gene_col].astype(str).str.split('.').str[0]
        
        # Remove version numbers from target_transcripts list
        target_tids_clean = [str(t).split('.')[0] for t in target_transcripts]
        
        # Create a set of gene IDs that correspond to our target transcripts
        valid_genes_mapping = mapping_df[mapping_df[tx_col].isin(target_tids_clean)]
        target_genes = set(valid_genes_mapping[gene_col].unique())
        
        # Keep only genes that exist in the TPM dataframe
        intersecting_genes = list(target_genes.intersection(set(tpm_df.index)))
        tpm_df = tpm_df.loc[intersecting_genes]
        print(f"Filtered TPM data to {len(intersecting_genes)} genes corresponding to target transcripts.")
        
        if len(intersecting_genes) < 2:
            raise ValueError("Not enough intersecting genes found between TPM matrix and Transcript evaluation list. Cannot compute correlation.")

    # If te_pivot is provided, further restrict TPM genes to those whose corresponding
    # transcripts are actually present (non-NaN) in the TE pivot. This ensures the
    # TPM correlation matrix is computed over the same feature subspace.
    if te_pivot is not None:
        # te_pivot index = transcript IDs, columns = cell types
        te_tids = set(str(t).split('.')[0] for t in te_pivot.index)
        # Map TE transcripts to genes
        mapping_df2 = pd.read_csv(mapping_file, sep='	')
        mapping_df2.columns = mapping_df2.columns.str.strip()
        tx_col2 = [c for c in mapping_df2.columns if 'Transcript' in c and 'ID' in c][0]
        gene_col2 = [c for c in mapping_df2.columns if 'Gene' in c and 'ID' in c][0]
        mapping_df2[tx_col2] = mapping_df2[tx_col2].astype(str).str.split('.').str[0]
        mapping_df2[gene_col2] = mapping_df2[gene_col2].astype(str).str.split('.').str[0]
        te_genes = set(mapping_df2[mapping_df2[tx_col2].isin(te_tids)][gene_col2].unique())
        # Intersect with current TPM genes
        shared_genes = list(te_genes.intersection(set(tpm_df.index)))
        if len(shared_genes) < 2:
            raise ValueError("Not enough shared genes between TE pivot and TPM matrix.")
        tpm_df = tpm_df.loc[shared_genes]
        print(f"After TE-pivot intersection: {len(shared_genes)} genes common to TPM, mapping, and TE pivot.")

    # Log1p transformation
    if log_transform:
        tpm_df = np.log1p(tpm_df)
        
    # Calculate cell-by-cell spearman correlation
    print("Computing TPM correlation matrix...")
    tpm_corr_matrix = tpm_df.corr(method='spearman')
    
    # Fill any remaining NaNs (e.g., if a cell has exactly zero variance across all filtered genes) with 0
    tpm_corr_matrix = tpm_corr_matrix.fillna(0)
    
    return tpm_corr_matrix


def align_matrices(mat1, mat2):
    """
    Ensure both correlation matrices have the exact same rows and columns in the same order.
    """
    common_cells = sorted(list(set(mat1.columns).intersection(set(mat2.columns))))
    if len(common_cells) < 3:
        raise ValueError("Not enough overlapping cells between the two matrices to compare.")
    
    # Subset and reorder both matrices
    mat1_aligned = mat1.loc[common_cells, common_cells]
    mat2_aligned = mat2.loc[common_cells, common_cells]
    
    return mat1_aligned, mat2_aligned


def evaluate_matrices_ari(mat1, mat2, n_clusters=3):
    """
    Cluster the cells based on the correlation matrices and compute the Adjusted Rand Index (ARI).
    Since correlation is a similarity measure, we convert it to distance (1 - corr).
    """
    mat1_aligned, mat2_aligned = align_matrices(mat1, mat2)
    
    # Convert correlation to distance (1 - correlation)
    dist1 = 1 - mat1_aligned.values
    dist2 = 1 - mat2_aligned.values
    
    # Ensure no negative distances due to float precision
    dist1 = np.clip(dist1, 0, None)
    dist2 = np.clip(dist2, 0, None)
    
    # Perform Hierarchical Clustering
    clusterer1 = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
    clusterer2 = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
    
    labels1 = clusterer1.fit_predict(dist1)
    labels2 = clusterer2.fit_predict(dist2)
    
    # Calculate Adjusted Rand Index
    ari_score = adjusted_rand_score(labels1, labels2)
    
    return ari_score


def evaluate_matrices_flat_correlation(mat1, mat2):
    """
    Extract the upper triangle of both correlation matrices and calculate 
    spearman R between them (similar to a Mantel test approach).
    """

    mat1_aligned, mat2_aligned = align_matrices(mat1, mat2)
    
    # Extract indices for the upper triangle (excluding diagonal)
    upper_tri_indices = np.triu_indices_from(mat1_aligned.values, k=1)
    
    # Extract values
    vals1 = mat1_aligned.values[upper_tri_indices]
    vals2 = mat2_aligned.values[upper_tri_indices]
    
    # Calculate spearman correlation
    r_val, p_val = spearmanr(vals1, vals2)
    
    return r_val, p_val