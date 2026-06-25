import os
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from collections import defaultdict
from plotnine import *
from scipy.stats import spearmanr
import warnings

from eval.calculate_te import *
from eval.multi_cell_te_specificity import (
    load_and_calculate_tpm_correlation,
    align_matrices,
    evaluate_matrices_flat_correlation,
    evaluate_matrices_ari
)

# 忽略警告
warnings.filterwarnings("ignore")

def extract_gt_and_cds_from_datasets(datasets):
    """
    Extract Ground Truth (GT) sequences and CDS info from single or multiple TranslationDatasets.
    Uses the dataset's own cell_type field (not uuid parsing).
    Supports 7-value unpack: uuid, species, cell_type, expr_vector, meta_info, seq_emb, count_emb.
    Returns: { cell_type: { tid: {'gt': [...], 'cds_start': x, 'cds_end': y, 'depth': z} } }
    """
    print("Extracting Ground Truth and CDS info from datasets...")
    info_dict = defaultdict(dict)

    if not isinstance(datasets, (list, tuple)):
        datasets = [datasets]

    for d_idx, dataset in enumerate(datasets):
        print(f"Parsing Dataset {d_idx + 1}/{len(datasets)}...")
        for i in tqdm(range(len(dataset)), desc=f"Dataset {d_idx + 1}"):
            # 7-value unpack from TranslationDataset.__getitem__
            uuid, species, cell_type, expr_vector, meta_info, seq_emb, count_emb = dataset[i]

            # Parse tid from uuid; cell_type comes directly from the dataset
            parts = str(uuid).rsplit('-', 2)
            if len(parts) < 2:
                continue
            tid = parts[0]
            # cell_type = parts[1]  # DO NOT overwrite—use the metadata cell_type from dataset

            if torch.is_tensor(count_emb):
                gt_array = count_emb.numpy().reshape(-1)
            else:
                gt_array = np.array(count_emb).reshape(-1)

            # Revert log1p transformation back to linear scale
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

        Args:
            datasets: a single TranslationDataset or a list/tuple of them
            pkl_paths: a single pkl path string or a list/tuple of them
        """
        self.datasets = datasets if isinstance(datasets, (list, tuple)) else [datasets]
        self.pkl_paths = pkl_paths if isinstance(pkl_paths, (list, tuple)) else [pkl_paths]
        self.min_depth = min_depth
        self.min_cells = min_cells

        self.dataset_info = extract_gt_and_cds_from_datasets(self.datasets)
        self.grouped_data = self._load_and_group_data()
        self.cell_types = self._get_all_cell_types()

        self.transcript_metrics_df = None
        self.pairwise_data = None
        self.te_pairwise_data = None
        self.te_pred_pivot = None
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
                        if cell_type not in self.dataset_info or tid not in self.dataset_info[cell_type]:
                            continue

                        ds_info = self.dataset_info[cell_type][tid]
                        cds_start = ds_info['cds_start']
                        cds_end = ds_info['cds_end']
                        gt_array = ds_info['gt']
                        depth = ds_info.get('rpf_depth', 0)

                        if cds_start < 0 or cds_end < 0:
                            continue

                        grouped[tid][cell_type] = {
                            'pred': np.expm1(np.array(pred_array).reshape(-1)),
                            'gt': gt_array,
                            'cds_start': int(cds_start) - 1,   # 1-based -> 0-based
                            'cds_end': int(cds_end) - 1,       # 1-based -> 0-based
                            'depth': float(depth)
                        }
                    except Exception:
                        continue

        print(f"Successfully integrated {len(self.pkl_paths)} files. Grouped into {len(grouped)} unique transcripts.")
        return grouped

    def _get_all_cell_types(self):
        cells = set()
        for t_data in self.grouped_data.values():
            cells.update(t_data.keys())
        return sorted(list(cells))

    # =========================================================
    # 核心方法：One-Pass Analysis
    # 一次性计算所有需要的矩阵，避免重复遍历
    # =========================================================
    def _run_global_analysis(self):
        if self._analysis_done: return

        print("Running global analysis (computing all matrices in one pass)...")
        
        # 容器初始化
        transcript_results = []
        # 使用 defaultdict 存储每对细胞的所有相关性值
        pair_stats = defaultdict(lambda: {'gt': [], 'pred': []})
        
        for tid, cells_dict in tqdm(self.grouped_data.items(), desc="Analyzing transcripts"):
            # 1. 过滤与准备数据
            available_cells = sorted(list(cells_dict.keys()))
            if len(available_cells) < self.min_cells: continue
            
            avg_depth = np.mean([cells_dict[c]['depth'] for c in available_cells])
            if avg_depth < self.min_depth: continue
            
            # 提取有效向量 (过滤方差极小的数据)
            valid_cells = []
            gt_vecs = []
            pred_vecs = []
            
            for c in available_cells:
                g = cells_dict[c]['gt']
                p = cells_dict[c]['pred']
                if np.std(g) > 1e-6 or np.std(p) > 1e-6:
                    valid_cells.append(c)
                    gt_vecs.append(g)
                    pred_vecs.append(p)
            
            if len(valid_cells) < 2: continue # 至少需要2个有效细胞
            
            # 堆叠矩阵 (N_cells x Length)
            gt_stack = np.stack(gt_vecs)
            pred_stack = np.stack(pred_vecs)
            
            # =========================================
            # 计算 1: GT 内部相关性 (GT vs GT)
            # 用途: Heatmap (Upper), Specificity (Bio_Sim)
            # =========================================
            mat_gt = np.corrcoef(gt_stack)
            
            # =========================================
            # 计算 2: Pred 内部相关性 (Pred vs Pred)
            # 用途: Heatmap (Lower)
            # =========================================
            mat_pred = np.corrcoef(pred_stack)
            
            # =========================================
            # 计算 3: Cross Correlation (Pred vs GT)
            # 用途: Specificity (Match vs Mismatch)
            # =========================================
            # 手动计算 Pred[i] vs GT[j] 矩阵
            p_mean = pred_stack.mean(axis=1, keepdims=True)
            g_mean = gt_stack.mean(axis=1, keepdims=True)
            p_centered = pred_stack - p_mean
            g_centered = gt_stack - g_mean
            
            numerator = p_centered @ g_centered.T # (N, L) @ (L, N) -> (N, N)
            p_std = np.sqrt((p_centered**2).sum(axis=1, keepdims=True))
            g_std = np.sqrt((g_centered**2).sum(axis=1, keepdims=True))
            denominator = p_std @ g_std.T
            
            mat_cross = numerator / (denominator + 1e-12)

            # =========================================
            # 数据分流 A: 存入 Transcript Metrics
            # =========================================
            # Bio Similarity (GT矩阵的上三角平均)
            upper_inds = np.triu_indices_from(mat_gt, k=1)
            bio_sim = np.nanmean(mat_gt[upper_inds])
            
            # Specificity (Cross矩阵: 对角线 vs 非对角线)
            r_match = np.nanmean(np.diag(mat_cross))
            
            mask_off = ~np.eye(len(valid_cells), dtype=bool)
            r_mismatch = np.nanmean(mat_cross[mask_off])
            
            transcript_results.append({
                'TID': tid,
                'N_Cells': len(valid_cells),
                'Avg_Depth': avg_depth,
                'Bio_Similarity': bio_sim,
                'R_Match': r_match,
                'R_Mismatch': r_mismatch,
                'Specificity_Score': r_match - r_mismatch
            })
            
            # =========================================
            # 数据分流 B: 存入 Pairwise Stats (用于 Heatmap)
            # =========================================
            n = len(valid_cells)
            for i in range(n):
                for j in range(i+1, n):
                    c1 = valid_cells[i]
                    c2 = valid_cells[j]
                    # 排序 key 保证一致性
                    if c1 > c2: c1, c2 = c2, c1
                    
                    # 收集 GT 相关性
                    if not np.isnan(mat_gt[i, j]):
                        pair_stats[(c1, c2)]['gt'].append(mat_gt[i, j])
                    
                    # 收集 Pred 相关性
                    if not np.isnan(mat_pred[i, j]):
                        pair_stats[(c1, c2)]['pred'].append(mat_pred[i, j])

        # --- 保存结果到 self ---
        self.transcript_metrics_df = pd.DataFrame(transcript_results).dropna()
        # 转换 float16 -> 32
        float16_cols = self.transcript_metrics_df.select_dtypes(include=['float16']).columns
        if len(float16_cols) > 0:
            self.transcript_metrics_df[float16_cols] = self.transcript_metrics_df[float16_cols].astype('float32')
            
        self.pairwise_data = pair_stats
        self._analysis_done = True
        print("Global analysis finished.")

    # =========================================================
    # Depth-correction analysis
    # =========================================================
    def plot_specificity_vs_depth(self, out_path="specificity_vs_depth.pdf"):
        """
        Scatter: transcript Specificity_Score vs log(Avg_Depth).
        A flat trendline indicates specificity is not driven by sequencing depth.
        """
        self._run_global_analysis()
        df = self.transcript_metrics_df.copy()
        df['Log_Depth'] = np.log1p(df['Avg_Depth'])
        clean = df.dropna(subset=['Log_Depth', 'Specificity_Score'])
        if len(clean) < 2:
            return

        r, p = spearmanr(clean['Log_Depth'], clean['Specificity_Score'])
        label = f"spearman R={r:.3f} (P={p:.2e})"

        p_plot = (
            ggplot(clean, aes(x='Log_Depth', y='Specificity_Score'))
            + geom_point(alpha=0.3, color="#2d3436", size=2, stroke=0)
            + geom_smooth(method='lm', color="#005b96", size=1)
            + annotate("text", x=clean["Log_Depth"].min(),
                       y=clean['Specificity_Score'].max(),
                       label=label, ha='left', va='top', size=10)
            + labs(x="log(Avg P-site depth + 1)",
                   y="Specificity Score (R_match - R_mismatch)")
            + theme_classic()
            + theme(figure_size=(4, 4))
        )
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        p_plot.save(out_path)
        print(f"Saved: {out_path}")
        return r, p

    def plot_specificity_vs_ncells(self, out_path="specificity_vs_ncells.pdf"):
        """
        Scatter: transcript Specificity_Score vs number of valid cell types.
        Checks whether specificity is inflated for transcripts seen in few cells.
        """
        self._run_global_analysis()
        df = self.transcript_metrics_df.copy()
        clean = df.dropna(subset=['N_Cells', 'Specificity_Score'])
        if len(clean) < 2:
            return

        r, p = spearmanr(clean['N_Cells'], clean['Specificity_Score'])
        label = f"spearman R={r:.3f} (P={p:.2e})"

        p_plot = (
            ggplot(clean, aes(x='N_Cells', y='Specificity_Score'))
            + geom_point(alpha=0.3, color="#2d3436", size=2, stroke=0)
            + geom_smooth(method='lm', color="#005b96", size=1)
            + annotate("text", x=clean["N_Cells"].max() * 0.6,
                       y=clean['Specificity_Score'].max(),
                       label=label, ha='left', va='top', size=10)
            + labs(x="Number of valid cell types",
                   y="Specificity Score (R_match - R_mismatch)")
            + theme_classic()
            + theme(figure_size=(4, 4))
        )
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        p_plot.save(out_path)
        print(f"Saved: {out_path}")
        return r, p

    def plot_biosim_vs_specificity(self, out_path="biosim_vs_specificity.pdf"):
        """
        Scatter: Bio_Similarity (mean GT inter-cell corr) vs Specificity_Score.
        Tests whether model specificity is higher for transcripts with
        inherently more variable translation patterns across cells.
        """
        self._run_global_analysis()
        df = self.transcript_metrics_df.copy()
        clean = df.dropna(subset=['Bio_Similarity', 'Specificity_Score'])
        if len(clean) < 2:
            return

        r, p = spearmanr(clean['Bio_Similarity'], clean['Specificity_Score'])
        label = f"spearman R={r:.3f} (P={p:.2e})"

        p_plot = (
            ggplot(clean, aes(x='Bio_Similarity', y='Specificity_Score'))
            + geom_point(alpha=0.3, color="#2d3436", size=2, stroke=0)
            + geom_smooth(method='lm', color="#005b96", size=1)
            + annotate("text", x=clean["Bio_Similarity"].min(),
                       y=clean['Specificity_Score'].max(),
                       label=label, ha='left', va='top', size=10)
            + labs(x="Bio Similarity (mean inter-cell GT corr)",
                   y="Specificity Score (R_match - R_mismatch)")
            + theme_classic()
            + theme(figure_size=(4, 4))
        )
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        p_plot.save(out_path)
        print(f"Saved: {out_path}")
        return r, p

    def export_transcript_metrics(self, out_dir="./results"):
        """Export per-transcript metrics CSV."""
        self._run_global_analysis()
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "transcript_specificity_metrics.csv")
        self.transcript_metrics_df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
        return self.transcript_metrics_df

    # ==========================================
    # 接口 1: 获取/保存 Specificity 结果 (deprecated, use export_transcript_metrics)
    # ==========================================

    # ==========================================
    # 接口 2: 获取/保存 Pairwise Heatmap 数据
    # ==========================================
    def compute_pairwise_matrices(self, out_dir="./results"):
        # 确保分析已运行
        self._run_global_analysis()
        
        os.makedirs(out_dir, exist_ok=True)
        
        records = []
        for (c1, c2), val_dict in self.pairwise_data.items():
            # 计算中位数
            gt_arr = np.array(val_dict['gt'], dtype=np.float32)
            pred_arr = np.array(val_dict['pred'], dtype=np.float32)
            
            median_gt = np.nanmedian(gt_arr) if len(gt_arr) > 0 else np.nan
            median_pred = np.nanmedian(pred_arr) if len(pred_arr) > 0 else np.nan
            
            records.append({
                "Cell 1": c1,
                "Cell 2": c2,
                "GT Corr": median_gt,
                "Pred Corr": median_pred
            })
            
        df = pd.DataFrame(records)
        csv_path = os.path.join(out_dir, "cell_type_pairwise_correlation.csv")
        df.to_csv(csv_path, index=False)
        print(f"Pairwise correlation table saved to {csv_path}")
        return df

    def extract_square_matrices_from_pairwise(self):
        """
        Reconstruct full square correlation DataFrames (GT and Pred)
        from the profile-based self.pairwise_data dictionary.
        """
        self._run_global_analysis()
        cells = self.cell_types

        gt_matrix = pd.DataFrame(1.0, index=cells, columns=cells)
        pred_matrix = pd.DataFrame(1.0, index=cells, columns=cells)

        for (c1, c2), val_dict in self.pairwise_data.items():
            if c1 == c2:
                continue
            gt_arr = np.array(val_dict['gt'], dtype=np.float32)
            pred_arr = np.array(val_dict['pred'], dtype=np.float32)
            gt_val = np.nanmedian(gt_arr) if len(gt_arr) > 0 else np.nan
            pred_val = np.nanmedian(pred_arr) if len(pred_arr) > 0 else np.nan

            gt_matrix.loc[c1, c2] = gt_val
            gt_matrix.loc[c2, c1] = gt_val
            pred_matrix.loc[c1, c2] = pred_val
            pred_matrix.loc[c2, c1] = pred_val

        print(f"Profile-based square matrices reconstructed: {len(cells)}x{len(cells)}")
        return pred_matrix, gt_matrix

    def compare_with_tpm(self, tpm_file, mapping_file, out_dir="./results"):
        """
        Compare profile-based GT/Pred cell-type correlation matrices with TPM.
        """
        self._run_global_analysis()
        pred_mat, gt_mat = self.extract_square_matrices_from_pairwise()
        # Also compute TE-based matrices for comparison
        self.compute_te_pairwise_matrices(out_dir=out_dir, log_transform=True)
        te_pred_mat, te_gt_mat = self.extract_te_square_matrices()

        target_cells = self.cell_types
        target_tids = list(self.grouped_data.keys())

        # Profile-based pivots (for gene intersection with TPM)
        # Build a pseudo-pivot: transcript x cell mean corr (we use TE pivot for gene-level matching)
        te_pivot = getattr(self, 'te_pred_pivot', None) if hasattr(self, 'te_pred_pivot') else None

        tpm_corr = load_and_calculate_tpm_correlation(
            tpm_file=tpm_file,
            mapping_file=mapping_file,
            target_cells=target_cells,
            target_transcripts=target_tids,
            te_pivot=te_pivot,
            log_transform=True
        )

        # Align all matrices to common cells
        common = sorted(list(
            set(gt_mat.columns) & set(pred_mat.columns) & set(tpm_corr.columns)
        ))
        if len(common) < 3:
            raise ValueError(f"Only {len(common)} common cells—not enough to compare.")

        gt_a = gt_mat.loc[common, common]
        pr_a = pred_mat.loc[common, common]
        tp_a = tpm_corr.loc[common, common]

        # Also align TE-based matrices
        te_pr_a, te_gt_a = align_matrices(te_pred_mat, te_gt_mat)
        te_pr_a, te_gt_a = te_pr_a.loc[common, common], te_gt_a.loc[common, common]

        sep = "=" * 60
        print()
        print(sep)
        print(f"Aligned on {len(common)} common cells")
        print(sep)

        # Profile-based
        r_pr_gt, p_pr_gt = evaluate_matrices_flat_correlation(pr_a, gt_a)
        r_pr_tpm, p_pr_tpm = evaluate_matrices_flat_correlation(pr_a, tp_a)
        r_gt_tpm, p_gt_tpm = evaluate_matrices_flat_correlation(gt_a, tp_a)
        ari_pr_gt = evaluate_matrices_ari(pr_a, gt_a)
        ari_gt_tpm = evaluate_matrices_ari(gt_a, tp_a)

        print()
        print("--- Profile-based correlation matrices ---")
        print(f"Pred vs GT:    spearman R={r_pr_gt:.4f} (p={p_pr_gt:.2e}), ARI={ari_pr_gt:.4f}")
        print(f"Pred vs TPM:   spearman R={r_pr_tpm:.4f} (p={p_pr_tpm:.2e})")
        print(f"GT vs TPM:     spearman R={r_gt_tpm:.4f} (p={p_gt_tpm:.2e}), ARI={ari_gt_tpm:.4f}")

        # TE-based
        r_te_pr_gt, p_te_pr_gt = evaluate_matrices_flat_correlation(te_pr_a, te_gt_a)
        r_te_pr_tpm, p_te_pr_tpm = evaluate_matrices_flat_correlation(te_pr_a, tp_a)
        r_te_gt_tpm, p_te_gt_tpm = evaluate_matrices_flat_correlation(te_gt_a, tp_a)

        print()
        print("--- TE-based correlation matrices ---")
        print(f"Pred vs GT:    spearman R={r_te_pr_gt:.4f} (p={p_te_pr_gt:.2e})")
        print(f"Pred vs TPM:   spearman R={r_te_pr_tpm:.4f} (p={p_te_pr_tpm:.2e})")
        print(f"GT vs TPM:     spearman R={r_te_gt_tpm:.4f} (p={p_te_gt_tpm:.2e})")

        return {
            "profile": {"pred_vs_gt": r_pr_gt, "pred_vs_tpm": r_pr_tpm, "gt_vs_tpm": r_gt_tpm},
            "te": {"pred_vs_gt": r_te_pr_gt, "pred_vs_tpm": r_te_pr_tpm, "gt_vs_tpm": r_te_gt_tpm},
            "n_cells": len(common)
        }

    def extract_te_square_matrices(self):
        """
        Reconstruct full square TE-based correlation DataFrames from self.te_pairwise_data.
        Mirrors multi_cell_te_specificity.extract_square_matrices_from_evaluator.
        """
        if not hasattr(self, 'te_pairwise_data') or self.te_pairwise_data is None:
            self.compute_te_pairwise_matrices()

        cells = self.cell_types
        gt_matrix = pd.DataFrame(1.0, index=cells, columns=cells)
        pred_matrix = pd.DataFrame(1.0, index=cells, columns=cells)

        for (c1, c2), metrics in self.te_pairwise_data.items():
            if c1 == c2:
                continue
            gt_val = metrics.get('GT', np.nan)
            pred_val = metrics.get('Pred', np.nan)
            gt_matrix.loc[c1, c2] = gt_val
            gt_matrix.loc[c2, c1] = gt_val
            pred_matrix.loc[c1, c2] = pred_val
            pred_matrix.loc[c2, c1] = pred_val

        return pred_matrix, gt_matrix

    # ==========================================
    # 绘图函数保持不变，它们只负责调用上面的接口
    # ==========================================


    def plot_merged_heatmap(self, out_path="merged_heatmap.pdf",
                            tpm_file=None, mapping_file=None):
        """
        Profile-based merged-triangle heatmap.
        Upper triangle = GT (observed).
        Lower triangle = Pred (model) by default, or TPM if tpm_file/mapping_file provided.
        """
        self._run_global_analysis()
        pairwise_df = self.compute_pairwise_matrices(out_dir=os.path.dirname(out_path) or ".")
        if len(pairwise_df) == 0:
            return

        # Build GT lookup from profile-based pairwise data
        lookup_gt = {}
        for _, row in pairwise_df.iterrows():
            c1, c2 = row['Cell 1'], row['Cell 2']
            lookup_gt[(c1, c2)] = row['GT Corr']
            lookup_gt[(c2, c1)] = row['GT Corr']

        # Build lower-triangle lookup: Pred or TPM
        if tpm_file is not None and mapping_file is not None:
            tpm_corr = self._load_tpm_for_heatmap(tpm_file, mapping_file)
            lookup_lower = {}
            cells_tpm = list(tpm_corr.columns)
            for i, c1 in enumerate(cells_tpm):
                for j, c2 in enumerate(cells_tpm):
                    if i != j:
                        lookup_lower[(c1, c2)] = tpm_corr.loc[c1, c2]
            lower_label = "TPM"
            title = "Profile: Obs. (Upper) vs TPM (Lower)"
        else:
            lookup_lower = {}
            for _, row in pairwise_df.iterrows():
                c1, c2 = row['Cell 1'], row['Cell 2']
                lookup_lower[(c1, c2)] = row['Pred Corr']
                lookup_lower[(c2, c1)] = row['Pred Corr']
            lower_label = "Pred"
            title = "Profile: Obs. (Upper) vs Pred. (Lower)"

        cells = self.cell_types
        plot_data = []
        for i, c1 in enumerate(cells):
            for j, c2 in enumerate(cells):
                if i == j:
                    val = 1.0
                elif i < j:
                    val = lookup_gt.get((c1, c2), np.nan)
                else:
                    val = lookup_lower.get((c1, c2), np.nan)
                plot_data.append({'Cell_X': c2, 'Cell_Y': c1, 'Correlation': val})

        df_plot = pd.DataFrame(plot_data)
        df_plot['Cell_X'] = pd.Categorical(df_plot['Cell_X'], categories=cells)
        df_plot['Cell_Y'] = pd.Categorical(df_plot['Cell_Y'], categories=list(reversed(cells)))

        p = (
            ggplot(df_plot, aes(x='Cell_X', y='Cell_Y', fill='Correlation'))
            + geom_tile(color="white", size=0.5)
            + scale_fill_distiller(palette="YlGnBu", direction=-1, limits=(0, 1))
            + labs(title=title, x="", y="")
            + theme_minimal()
            + theme(axis_text_x=element_text(rotation=45, hjust=1),
                    figure_size=(7, 6), panel_grid=element_blank())
        )
        p.save(out_path)
        print(f"Saved: {out_path}")

    def compute_te_pairwise_matrices(self, out_dir="./results", log_transform=True):
        """
        提取所有转录本的 TE，并计算细胞类型之间的 TE 相关性矩阵
        """
        print("Calculating Transcript TE and pairwise correlations...")
        os.makedirs(out_dir, exist_ok=True)
        
        # 1. 提取所有的 TE 值
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
        
        # [修复]：强制将 TE 数据转换为 float32，解决 Pandas Cython 底层不支持 float16 导致的 TypeError
        te_df['GT_TE'] = te_df['GT_TE'].astype('float32')
        te_df['Pred_TE'] = te_df['Pred_TE'].astype('float32')
        
        # 顺手保存一下所有转录本的 TE 原始计算值
        te_df.to_csv(os.path.join(out_dir, "transcript_TE_values.csv"), index=False)
        
        # 2. 将长表透视为宽表 (Rows: TID, Columns: Cell Types)
        gt_pivot = te_df.pivot_table(index='TID', columns='Cell', values='GT_TE', aggfunc='mean')
        pred_pivot = te_df.pivot_table(index='TID', columns='Cell', values='Pred_TE', aggfunc='mean')

        # Align GT and Pred pivots: keep only (tid, cell) pairs present in BOTH
        common_mask = gt_pivot.notna() & pred_pivot.notna()
        gt_pivot = gt_pivot.where(common_mask)
        pred_pivot = pred_pivot.where(common_mask)
        valid_rows = common_mask.any(axis=1)
        gt_pivot = gt_pivot.loc[valid_rows]
        pred_pivot = pred_pivot.loc[valid_rows]
        print(f"Aligned TE pivot: {gt_pivot.shape[0]} transcripts x {gt_pivot.shape[1]} cells (shared mask)")

        # Store for TPM intersection
        self.te_pred_pivot = pred_pivot

        # 3. 对数转换 (防止极高表达的管家基因主导 spearman 相关性)
        if log_transform:
            gt_pivot = np.log1p(gt_pivot)
            pred_pivot = np.log1p(pred_pivot)
            
        # 4. 计算相关性矩阵 (min_periods avoids spurious high corr from too few transcripts)
        gt_corr_matrix = gt_pivot.corr(method="spearman", min_periods=5)
        pred_corr_matrix = pred_pivot.corr(method="spearman", min_periods=5)
        
        # 5. 格式化并缓存，用于热图和 CSV
        cells = self.cell_types
        records = []
        self.te_pairwise_data = {} 
        
        for i, c1 in enumerate(cells):
            for j, c2 in enumerate(cells):
                # 检查是否在矩阵列中
                if c1 in gt_corr_matrix.columns and c2 in gt_corr_matrix.columns:
                    val_gt = gt_corr_matrix.loc[c1, c2]
                    val_pred = pred_corr_matrix.loc[c1, c2]
                else:
                    val_gt, val_pred = np.nan, np.nan
                    
                # 缓存供绘图使用 (包含双向)
                self.te_pairwise_data[(c1, c2)] = {'GT': val_gt, 'Pred': val_pred}
                
                # 只保留唯一 pair 存入 CSV
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

    def plot_te_merged_heatmap(self, out_path="te_merged_heatmap.pdf",
                               tpm_file=None, mapping_file=None):
        """
        TE-based merged-triangle heatmap.
        Upper triangle = GT (observed).
        Lower triangle = Pred (model) by default, or TPM if tpm_file/mapping_file provided.
        """
        if not hasattr(self, 'te_pairwise_data') or self.te_pairwise_data is None:
            self.compute_te_pairwise_matrices(out_dir=os.path.dirname(out_path) or ".", log_transform=True)

        cells = self.cell_types

        # Build lower-triangle lookup: Pred or TPM
        if tpm_file is not None and mapping_file is not None:
            tpm_corr = self._load_tpm_for_heatmap(tpm_file, mapping_file)
            lookup_lower = {}
            cells_tpm = list(tpm_corr.columns)
            for i, c1 in enumerate(cells_tpm):
                for j, c2 in enumerate(cells_tpm):
                    if i != j:
                        lookup_lower[(c1, c2)] = tpm_corr.loc[c1, c2]
            lower_label = "TPM"
            title = "TE: Obs. (Upper) vs TPM (Lower)"
        else:
            lookup_lower = None  # use self.te_pairwise_data['Pred']
            title = "TE: Obs. (Upper) vs Pred. (Lower)"

        plot_data = []
        for i, c1 in enumerate(cells):
            for j, c2 in enumerate(cells):
                if i == j:
                    val = 1.0
                elif i < j:
                    val = self.te_pairwise_data.get((c1, c2), {}).get('GT', np.nan)
                else:
                    if lookup_lower is not None:
                        val = lookup_lower.get((c1, c2), np.nan)
                    else:
                        val = self.te_pairwise_data.get((c1, c2), {}).get('Pred', np.nan)
                plot_data.append({'Cell_X': c2, 'Cell_Y': c1, 'Correlation': val})

        df_plot = pd.DataFrame(plot_data)
        df_plot['Cell_X'] = pd.Categorical(df_plot['Cell_X'], categories=cells)
        df_plot['Cell_Y'] = pd.Categorical(df_plot['Cell_Y'], categories=list(reversed(cells)))

        p = (
            ggplot(df_plot, aes(x='Cell_X', y='Cell_Y', fill='Correlation'))
            + geom_tile(color="white", size=0.5)
            + scale_fill_distiller(palette="OrRd", direction=-1, limits=(0, 1))
            + labs(title=title, x="", y="")
            + theme_minimal()
            + theme(axis_text_x=element_text(rotation=45, hjust=1),
                    figure_size=(7, 6), panel_grid=element_blank())
        )
        p.save(out_path)
        print(f"Saved: {out_path}")

    def _load_tpm_for_heatmap(self, tpm_file, mapping_file):
        """Load TPM correlation matrix aligned to evaluator cells/transcripts."""
        if self.te_pred_pivot is None:
            self.compute_te_pairwise_matrices(log_transform=True)
        return load_and_calculate_tpm_correlation(
            tpm_file=tpm_file,
            mapping_file=mapping_file,
            target_cells=self.cell_types,
            target_transcripts=list(self.grouped_data.keys()),
            te_pivot=getattr(self, 'te_pred_pivot', None),
            log_transform=True,
        )


