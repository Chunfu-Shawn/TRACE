import os
import pickle
import warnings
import numpy as np
import pandas as pd
from typing import Optional
from plotnine import *
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# =================================================================
# [NEW] 定义全局配置：统一的颜色与顺序
# =================================================================
GLOBAL_MODEL_COLORS = {
    "TRACE": "#2C6B9A",
    "Convolution": "#637D96",
    "TranslationAI": "#555555",
    "RiboTIE": "#777777",
    "RibORF": "#BBBBBB",
    "RiboTISH": "#999999",
    "ORF-structure": "#AF804F",
    "Transcription-level": "#EBC67F"
}

GLOBAL_MODEL_ORDER = [
    "TRACE", 
    "Convolution", 
    "TranslationAI", 
    "RiboTIE", 
    "RiboTISH", 
    "RibORF",
    "ORF-structure", 
    "Transcription-level"
]


def plot_model_benchmark(
        manifest: list, 
        out_dir: str = "./results/benchmark",
        depth_levels: list = ['1M', '2M', '5M', '10M', 'Total']
):
    """
    一次性读取多个模型的评估结果 CSV，绘制 Ribo-seq 深度与 AUC 的趋势对比图。
    """
    os.makedirs(out_dir, exist_ok=True)
    print("Loading and aggregating AUC benchmark data...")
    
    records = []
    
    def extract_auc_metrics(df, target_feature):
        if target_feature and 'Feature' in df.columns:
            sub_df = df[df['Feature'] == target_feature]
            if sub_df.empty: return None, None
            row = sub_df.iloc[0]
        else:
            row = df.sort_values(by='PR-AUC', ascending=False).iloc[0]
        return row['ROC-AUC'], row['PR-AUC']
        
    for cfg in manifest:
        model_name = cfg['model']
        model_type = cfg['type']  
        target_feature = cfg.get('feature', None)
        
        if model_type == 'w/o Ribo-seq':
            csv_path = cfg['path']
            if not os.path.exists(csv_path):
                continue
                
            df = pd.read_csv(csv_path)
            roc_auc, pr_auc = extract_auc_metrics(df, target_feature)
            
            if roc_auc is not None and pr_auc is not None:
                for d in depth_levels:
                    records.append({
                        'Model': model_name, 'Type': model_type, 'Depth': d,
                        'ROC-AUC': roc_auc, 'PR-AUC': pr_auc
                    })
                
        elif model_type == 'w/ Ribo-seq':
            base_dir = cfg['base_dir']
            file_name = cfg.get('file_name', 'overall_metrics.csv')
            target_depths = cfg.get('depths', depth_levels)
            
            for d in target_depths:
                csv_path = os.path.join(base_dir, d, file_name)
                if not os.path.exists(csv_path):
                    continue
                    
                df = pd.read_csv(csv_path)
                roc_auc, pr_auc = extract_auc_metrics(df, target_feature)
                
                if roc_auc is not None and pr_auc is not None:
                    records.append({
                        'Model': model_name, 'Type': model_type, 'Depth': d,
                        'ROC-AUC': roc_auc, 'PR-AUC': pr_auc
                    })
            
    if not records:
        raise ValueError("No valid records extracted.")
        
    plot_df = pd.DataFrame(records)
    plot_df['Depth'] = pd.Categorical(plot_df['Depth'], categories=depth_levels, ordered=True)

    print("Generating Benchmark Trend Plots...")
    
    # =================================================================
    # [MODIFIED] 动态过滤类别顺序，只保留数据中存在的模型
    # =================================================================
    actual_models = plot_df['Model'].unique().tolist()
    # 按照 GLOBAL_MODEL_ORDER 的顺序提取实际存在的模型
    valid_order = [m for m in GLOBAL_MODEL_ORDER if m in actual_models]
    # 处理未在配置表中声明的新模型（追加在末尾）
    for m in actual_models:
        if m not in valid_order:
            valid_order.append(m)
            
    plot_df['Model'] = pd.Categorical(plot_df['Model'], categories=valid_order, ordered=True)

    color_mapping = {m: GLOBAL_MODEL_COLORS.get(m, "#C0C0C0") for m in valid_order}
    
    def build_trend_plot(metric_name: str, y_label: str):
        p = (
            ggplot(plot_df, aes(x='Depth', y=metric_name, color='Model', group='Model'))
            + geom_line(aes(linetype='Type'), size=1.5, alpha=0.8)
            + geom_point(data=plot_df[plot_df['Type'] == 'w/ Ribo-seq'], size=3.5, alpha=0.9)
            + scale_color_manual(values=color_mapping)
            + scale_linetype_manual(values={'w/ Ribo-seq': 'dashed', 'w/o Ribo-seq': 'solid'})
            + scale_x_discrete(expand=[0, 0])
            + theme_bw()
            + labs(x="Ribo-seq Data Depth", y=y_label)
            + theme(
                panel_border=element_rect(color="black", size=1),
                axis_title=element_text(size=12),
                axis_text_x=element_text(rotation=0, ha='center', size=10),
                axis_text_y=element_text(size=10),
                legend_position="right",
                legend_title=element_blank()
            )
        )
        return p

    p_roc = build_trend_plot('ROC-AUC', 'ROC-AUC')
    p_roc.save(os.path.join(out_dir, "Benchmark_ROC_AUC_Trend.pdf"), dpi=300, verbose=False)
    p_pr = build_trend_plot('PR-AUC', 'PR-AUC')
    p_pr.save(os.path.join(out_dir, "Benchmark_PR_AUC_Trend.pdf"), dpi=300, verbose=False)
    print(f"✅ Benchmark Complete! Plots saved to: {out_dir}")


def plot_tradeoff_benchmark(
        manifest: list, 
        out_dir: str = "./results/benchmark",
        depth_levels: list = ['1M', '5M', '10M', '50M', '100M', 'Total'],
        x_col: str = 'TP_at_Best_Threshold',
        y_col: str = 'Best_F1_Score',
        x_label: str = 'True Positives at Best F1 (Log Scale)',
        y_label: str = 'Best F1-Score',
        title: str = 'Quantity-Quality Trade-off at Best Threshold'
):
    os.makedirs(out_dir, exist_ok=True)
    records = []
    
    for cfg in manifest:
        model_name = cfg['model']
        model_type = cfg['type']  
        
        if model_type == 'w/o Ribo-seq':
            csv_path = cfg['path']
            if not os.path.exists(csv_path): continue
            df = pd.read_csv(csv_path)
            records.append({
                'Model': model_name, 'Type': model_type, 'Depth': 'Constant', 
                x_col: df.iloc[0][x_col], y_col: df.iloc[0][y_col]
            })
            
        elif model_type == 'w/ Ribo-seq':
            base_dir = cfg['base_dir']
            file_name = cfg.get('file_name', 'overall_prediction_summary.csv')
            target_depths = cfg.get('depths', depth_levels)
            
            for d in target_depths:
                csv_path = os.path.join(base_dir, d, file_name)
                if not os.path.exists(csv_path): continue
                df = pd.read_csv(csv_path)
                records.append({
                    'Model': model_name, 'Type': model_type, 'Depth': d,
                    x_col: df.iloc[0][x_col], y_col: df.iloc[0][y_col]
                })
            
    if not records: raise ValueError("No valid records extracted.")
        
    plot_df = pd.DataFrame(records)
    all_depth_categories = depth_levels + ['Constant']
    plot_df['Depth'] = pd.Categorical(plot_df['Depth'], categories=all_depth_categories, ordered=True)

    # =================================================================
    # [MODIFIED] 动态过滤类别顺序
    # =================================================================
    actual_models = plot_df['Model'].unique().tolist()
    valid_order = [m for m in GLOBAL_MODEL_ORDER if m in actual_models]
    for m in actual_models:
        if m not in valid_order: valid_order.append(m)
            
    plot_df['Model'] = pd.Categorical(plot_df['Model'], categories=valid_order, ordered=True)
    color_mapping = {m: GLOBAL_MODEL_COLORS.get(m, "#C0C0C0") for m in valid_order}
            
    min_size, max_size = 2, 5  
    depth_sizes = np.linspace(min_size, max_size, len(depth_levels))
    size_mapping = {d: s for d, s in zip(depth_levels, depth_sizes)}
    size_mapping['Constant'] = 5 

    p = (
        ggplot(plot_df, aes(x=x_col, y=y_col, color='Model'))
        + geom_line(data=plot_df[plot_df['Type'] == 'w/ Ribo-seq'], mapping=aes(group='Model'), linetype='dashed', size=1.2, alpha=0.7)
        + geom_point(mapping=aes(size='Depth'), alpha=0.9, stroke=0.5)
        + scale_x_log10()
        + scale_color_manual(values=color_mapping)
        + scale_size_manual(values=size_mapping, breaks=depth_levels, name="Ribo-seq Depth")
        + theme_bw()
        + labs(title=title, x=x_label, y=y_label)
        + theme(
            panel_border=element_rect(color="black", size=1),
            legend_position="right",
            legend_title=element_text(size=10, face="bold") 
        )
    )
    save_path = os.path.join(out_dir, f"Benchmark_Tradeoff_{x_col}_vs_{y_col}.pdf")
    p.save(save_path, dpi=300, verbose=False)


def plot_multi_model_top_k_precision(
        manifest: list, 
        out_dir: str = "./results/benchmark", 
        min_k: Optional[int] = None, 
        max_k: Optional[int] = None, 
        suffix: str = ""
):
    os.makedirs(out_dir, exist_ok=True)
    all_pk_data = []
    
    for cfg in manifest:
        model_name = cfg['model']
        csv_path = cfg['path']
        score_col = cfg.get('score_col', 'score') 
        
        if not os.path.exists(csv_path): continue
        df = pd.read_csv(csv_path)
        
        if 'Precision' in df.columns and 'K' in df.columns:
            pk_df = df[['K', 'Precision']].copy()
        elif 'y_true' in df.columns and score_col in df.columns:
            df_sorted = df.sort_values(by=score_col, ascending=False).reset_index(drop=True)
            df_sorted = df_sorted[df_sorted[score_col] >= 0].copy()
            if df_sorted.empty: continue
            k_array = np.arange(1, len(df_sorted) + 1)
            tp_cumsum = df_sorted['y_true'].cumsum()
            pk_df = pd.DataFrame({'K': k_array, 'Precision': tp_cumsum / k_array})
        else: continue
            
        pk_df['Model'] = model_name
        all_pk_data.append(pk_df)
        
    if not all_pk_data: raise ValueError("No valid Top-K data processed.")
    plot_df = pd.concat(all_pk_data, ignore_index=True)
        
    def apply_smoothing(group):
        group['Precision_Smooth'] = group['Precision'].rolling(window=50, min_periods=1).mean()
        return group
        
    plot_df = plot_df.groupby('Model', group_keys=False).apply(apply_smoothing)

    if min_k is not None: plot_df = plot_df[plot_df['K'] >= min_k]
    if max_k is not None: plot_df = plot_df[plot_df['K'] <= max_k]

    def downsample(group, max_pts=3000):
        if len(group) > max_pts:
            indices = np.linspace(0, len(group) - 1, max_pts).astype(int)
            return group.iloc[indices]
        return group
    plot_df = plot_df.groupby('Model', group_keys=False).apply(downsample)

    # =================================================================
    # [MODIFIED] 动态过滤类别顺序
    # =================================================================
    actual_models = plot_df['Model'].unique().tolist()
    valid_order = [m for m in GLOBAL_MODEL_ORDER if m in actual_models]
    for m in actual_models:
        if m not in valid_order: valid_order.append(m)
            
    plot_df['Model'] = pd.Categorical(plot_df['Model'], categories=valid_order, ordered=True)

    color_mapping = {m: GLOBAL_MODEL_COLORS.get(m, "#C0C0C0") for m in valid_order}
    linetype_mapping = {m: "solid" if "TRACE" in m else "dashed" for m in valid_order}

    if min_k is not None and max_k is not None:
        title_suffix = f"(K: {min_k} to {max_k})"
        file_suffix = f"{suffix}_{min_k}_to_{max_k}"
    elif min_k is not None:
        title_suffix = f"(K >= {min_k})"
        file_suffix = f"{suffix}_{min_k}_to_All"
    elif max_k is not None:
        title_suffix = f"(Top {max_k})"
        file_suffix = f"{suffix}_1_to_{max_k}"
    else:
        title_suffix = "(All Predictions)"
        file_suffix = f"{suffix}_All"

    p = (
        ggplot(plot_df, aes(x='K', y='Precision_Smooth', color='Model'))
        + geom_line(aes(linetype='Model'), size=1.5, alpha=0.85)
        + scale_color_manual(values=color_mapping)
        + scale_linetype_manual(values=linetype_mapping, guide=None)
        + scale_y_continuous(limits=(0, 1.05))
        + scale_x_log10() 
        + theme_classic()
        + labs(
            title=f"Precision@K Benchmark {title_suffix}",
            x="Top K Predicted ORFs (Log Scale, Ranked by Conf. Score)",
            y="Precision (Proportion of True Positives)"
        )
        + theme(
            figure_size=(7, 5),
            axis_title=element_text(size=12, face="bold"),
            axis_text=element_text(size=10),
            legend_position="right",
            legend_text=element_text(size=10),
            legend_title=element_blank()
        )
    )
    p.save(os.path.join(out_dir, f"Benchmark_TopK_Precision_Curve_{file_suffix}.pdf"), dpi=300, verbose=False)


# 忽略全零向量计算相关性时的警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

def calculate_spearman(gt_signal: np.ndarray, pred_signal: np.ndarray) -> float:
    """计算两条等长信号的 Spearman 相关系数"""
    # 长度不一致、太短，或者其中一条毫无波澜(方差为0)，都无法计算相关性
    if len(gt_signal) != len(pred_signal) or len(gt_signal) < 3:
        return np.nan
    if np.std(gt_signal) < 1e-6 or np.std(pred_signal) < 1e-6:
        return np.nan
        
    r_val, _ = spearmanr(gt_signal, pred_signal)
    return float(r_val)


# ==============================================================================
# Step 1: Data Extraction & Correlation Calculation
# ==============================================================================
def extract_and_rank_correlation(
    file_config: dict, 
    gt_name: str = "Observation",
    target_cell: str = None,
    min_read_density: float = 0.1   # 转录本平均 Read 密度阈值
) -> pd.DataFrame:
    """
    加载 PKL 文件，提取真实数据的 Read 密度并据此分配 Rank。
    然后计算所有模型与真实值之间的 Position-wise Spearman Correlation。
    """
    loaded_data = {}
    print(f"--- [Step 1] Loading Data & Calculating Correlation ---")
    
    # 1. 加载所有字典
    for model_name, pkl_path in file_config.items():
        if not os.path.exists(pkl_path):
            print(f"  [Error] File not found for {model_name}: {pkl_path}")
            continue
            
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            
        is_nested = isinstance(data, dict) and any(isinstance(v, dict) for v in data.values())
        if is_nested:
            if target_cell and target_cell in data:
                loaded_data[model_name] = data[target_cell]
            else:
                fallback_cell = list(data.keys())[0]
                loaded_data[model_name] = data[fallback_cell]
                print(f"  [Warning] {model_name}: Auto-fallback to cell '{fallback_cell}'")
        else:
            loaded_data[model_name] = data

    if gt_name not in loaded_data:
        raise ValueError(f"Ground Truth key '{gt_name}' not found.")

    # 2. 从 GT 中提取高置信度转录本，并计算密度排序
    gt_dict = loaded_data.pop(gt_name) # 把 GT 抽出来单独作为标尺
    gt_densities = {}
    
    for tid, val in gt_dict.items():
        signal = np.asarray(val, dtype=np.float32)
        tx_len = len(signal)
        total_reads = np.sum(signal)
        
        if tx_len > 0 and (total_reads / tx_len) > min_read_density:
            gt_densities[tid] = total_reads / tx_len

    # 按密度从大到小降序排列
    sorted_tids = sorted(gt_densities.keys(), key=lambda x: gt_densities[x], reverse=True)
    tid_to_rank = {tid: rank for rank, tid in enumerate(sorted_tids)}
    
    print(f"  -> Ground Truth Filter: Retained {len(sorted_tids)} transcripts (Density > {min_read_density}).")

    # 3. 计算各个模型针对这些转录本的相关性
    records = []
    for model_name, model_dict in loaded_data.items():
        valid_count = 0
        for tid in sorted_tids:
            if tid in model_dict:
                pred_signal = np.asarray(model_dict[tid], dtype=np.float32)
                gt_signal = np.asarray(gt_dict[tid], dtype=np.float32)
                
                corr = calculate_spearman(gt_signal, pred_signal)
                if not np.isnan(corr):
                    records.append({
                        "Tid": tid,
                        "Rank": tid_to_rank[tid],
                        "Model": model_name,
                        "Correlation": corr
                    })
                    valid_count += 1
                    
        print(f"  -> {model_name}: Successfully computed correlation for {valid_count} transcripts.")

    df_ranked = pd.DataFrame(records)
    print("✅ Correlation ranking complete. Ready for plotting.")
    return df_ranked


def plot_top_k_precision_bar(
        manifest: list, target_k: int,
        out_dir: str = "./results/benchmark", 
        suffix: str = ""
):
    os.makedirs(out_dir, exist_ok=True)
    records = []
    
    for cfg in manifest:
        model_name, csv_path = cfg['model'], cfg['path']
        if not os.path.exists(csv_path): continue
        df = pd.read_csv(csv_path)
        prec_val = np.nan
        
        if 'Precision' in df.columns and 'K' in df.columns:
            if target_k in df['K'].values: prec_val = df.loc[df['K'] == target_k, 'Precision'].values[0]
            else:
                max_k_avail = df['K'].max()
                if 'TP_Count' in df.columns: prec_val = df.loc[df['K'] == max_k_avail, 'TP_Count'].values[0] / target_k
                else: prec_val = (df['Precision'].iloc[-1] * max_k_avail) / target_k
        elif 'y_true' in df.columns and cfg.get('score_col', 'score') in df.columns:
            df_sorted = df.sort_values(by=cfg.get('score_col', 'score'), ascending=False)
            df_sorted = df_sorted[df_sorted[cfg.get('score_col', 'score')] >= 0].copy()
            if df_sorted.empty: prec_val = 0.0
            else: prec_val = df_sorted['y_true'].iloc[:target_k].sum() / target_k if len(df_sorted) >= target_k else df_sorted['y_true'].sum() / target_k
        else: continue
            
        records.append({'Model': model_name, 'Dataset': cfg.get('dataset', 'Unknown'), 'Cell_type': cfg.get('cell_type', 'Unknown'), 'Precision': prec_val})
        
    if not records: raise ValueError("No valid Top-K data processed.")
    plot_df = pd.DataFrame(records)

    summary_df = plot_df.groupby('Model', observed=False).agg(
        Overall_Mean=('Precision', 'mean'),
        SEM=('Precision', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0)
    ).reset_index()
    summary_df['ymin'] = summary_df['Overall_Mean'] - summary_df['SEM']
    summary_df['ymax'] = summary_df['Overall_Mean'] + summary_df['SEM']

    # =================================================================
    # [MODIFIED] 动态过滤类别顺序
    # =================================================================
    actual_models = plot_df['Model'].unique().tolist()
    valid_order = [m for m in GLOBAL_MODEL_ORDER if m in actual_models]
    for m in actual_models:
        if m not in valid_order: valid_order.append(m)
            
    plot_df['Model'] = pd.Categorical(plot_df['Model'], categories=valid_order, ordered=True)
    summary_df['Model'] = pd.Categorical(summary_df['Model'], categories=valid_order, ordered=True)
    model_colors = {m: GLOBAL_MODEL_COLORS.get(m, "#C0C0C0") for m in valid_order}
            
    unique_cells = plot_df['Cell_type'].unique().tolist()
    unseen_cells = [c for c in unique_cells if 'unseen' in str(c).lower()]
    ordered_cells = [c for c in unique_cells if c not in unseen_cells] + unseen_cells
    plot_df['Cell_type'] = pd.Categorical(plot_df['Cell_type'], categories=ordered_cells, ordered=True)
    cell_colors = {ct: "#D6715E" if ct in unseen_cells else "#202020" for ct in ordered_cells}

    unique_datasets = plot_df['Dataset'].unique().tolist()
    plot_df['Dataset'] = pd.Categorical(plot_df['Dataset'], categories=unique_datasets, ordered=True)
    dataset_shapes = {ds: ['o', '^', 's', 'D', 'v', 'p', 'h', '8'][i % 8] for i, ds in enumerate(unique_datasets)}

    p = (
        ggplot()
        + geom_col(data=summary_df, mapping=aes(x='Model', y='Overall_Mean', fill='Model'), width=0.7)
        + geom_errorbar(data=summary_df, mapping=aes(x='Model', ymin='ymin', ymax='ymax'), width=0.2, size=0.8)
        + geom_jitter(data=plot_df, mapping=aes(x='Model', y='Precision', shape='Dataset', color='Cell_type'), width=0.15, size=3.0)
        + scale_fill_manual(values=model_colors, guide=None) 
        + scale_shape_manual(values=dataset_shapes, name="Dataset") 
        + scale_color_manual(values=cell_colors, name="Cell type")
        + theme_bw() 
        + labs(x="", y=f"Precision @ K={target_k}")
        + theme(axis_text_x=element_text(angle=45, hjust=1), legend_position="right")
    )
    p.save(os.path.join(out_dir, f"precision_at_{target_k}_bar{suffix}.pdf"), dpi=300, verbose=False)
    return summary_df, plot_df