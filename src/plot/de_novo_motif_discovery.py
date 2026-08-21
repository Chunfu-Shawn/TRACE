"""Plotting utilities for de novo motif and positional analyses."""

import os

import logomaker
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIXED_CDS_LEN = 600

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"


def _as_pdf_path(path):
    """Return an output path with a PDF suffix."""
    return f"{os.path.splitext(os.fspath(path))[0]}.pdf"


def _assign_frame_colors(df):
    """
    Since the metagene coordinate x_pos intrinsically preserves the frame,
    we can safely calculate frame directly from x_pos.
    Frame 0: Red (#E41A1C), Frame 1: Blue (#377EB8), Frame 2: Gray (gray)
    """
    df['frame'] = df['x_pos'].astype(int) % 3
    color_map = {0: '#E41A1C', 1: '#377EB8', 2: 'gray'}
    df['frame_color'] = df['frame'].map(color_map)
    df['Frame'] = df['frame'].map({0: 'Frame 0', 1: 'Frame 1', 2: 'Frame 2'})
    df['Frame'] = pd.Categorical(df['Frame'], categories=['Frame 0', 'Frame 1', 'Frame 2'])
    return df


def _cds_rect_data():
    """Build geom_rect data for a single continuous CDS shading."""
    return pd.DataFrame({
        'xmin': [0], 
        'xmax': [FIXED_CDS_LEN], 
        'ymin': [-float('inf')], 
        'ymax': [float('inf')], 
        'fill': ['lightgray']
    })


def plot_attention_profile(attn_df, out_path="attention_profile.pdf", up_len=300, down_len=300, 
                           color_by_frame=True, xlim=None, show_xaxis=False, show_cds=True, 
                           weight=6, height=5):
    
    from plotnine import (ggplot, aes, geom_line, geom_point, geom_rect,
                          labs, theme, facet_grid, scale_color_manual, scale_fill_identity,
                          element_text, theme_classic, element_blank, element_line)
    
    # 1. Bounds filtering (support explicit xlim for zooming in)
    if xlim is not None:
        df_plot = attn_df[(attn_df['x_pos'] >= xlim[0]) & (attn_df['x_pos'] <= xlim[1])].copy()
    else:
        df_plot = attn_df[(attn_df['x_pos'] >= -up_len) & (attn_df['x_pos'] <= FIXED_CDS_LEN + down_len - 1)].copy()
        
    if df_plot.empty:
        raise ValueError("No attention positions remain within the plot range.")
    df_plot['layer'] = df_plot['layer'].astype(int)
    has_head_profiles = 'head' in df_plot.columns
    if has_head_profiles:
        df_plot['head'] = df_plot['head'].astype(int)

    # 2. Aggregation logic based on whether we group by Frame
    if color_by_frame:
        df_plot = _assign_frame_colors(df_plot)
        group_cols = ['layer', 'x_pos', 'Frame']
        head_group_cols = ['layer', 'head', 'x_pos', 'Frame']
    else:
        group_cols = ['layer', 'x_pos']
        head_group_cols = ['layer', 'head', 'x_pos']

    if has_head_profiles:
        df_head_plot = df_plot.groupby(
            head_group_cols, as_index=False, observed=True
        )[['mean_attn']].mean().dropna(subset=['mean_attn'])
        df_head_plot['log2_mean_attn'] = np.log2(
            df_head_plot['mean_attn'] + 1
        )
    else:
        df_head_plot = pd.DataFrame()

    df_plot = df_plot.groupby(group_cols, as_index=False, observed=True)[['mean_attn']].mean().dropna(subset=['mean_attn'])
    df_plot['log2_mean_attn'] = np.log2(df_plot['mean_attn'] + 1)

    base_out = os.path.splitext(os.fspath(out_path))[0]
    rect_cds = _cds_rect_data()
    
    # Dynamic axis styling based on show_xaxis
    x_axis_text = element_text() if show_xaxis else element_blank()
    x_axis_ticks = element_line() if show_xaxis else element_blank()
    x_axis_title = element_text() if show_xaxis else element_blank()
    x_label_str = 'Metagene Position (x_pos)' if show_xaxis else ''

    # ==================================
    # Combined Plot
    # ==================================
    if color_by_frame:
        df_combined = df_plot.groupby(['x_pos', 'Frame'], as_index=False, observed=True)[['mean_attn']].mean()
    else:
        df_combined = df_plot.groupby(['x_pos'], as_index=False, observed=True)[['mean_attn']].mean()
        
    df_combined['log2_mean_attn'] = np.log2(df_combined['mean_attn'] + 1)

    p_comb = (
        ggplot(df_combined, aes(x='x_pos', y='log2_mean_attn'))
        + scale_fill_identity()
    )

    if show_cds:
        p_comb += geom_rect(data=rect_cds, mapping=aes(xmin='xmin', xmax='xmax', ymin='ymin', ymax='ymax', fill='fill'), alpha=0.3, inherit_aes=False, show_legend=False)

    if color_by_frame:
        frame_palette = {'Frame 0': '#D73027', 'Frame 1': '#4575B4', 'Frame 2': 'darkgray'}
        p_comb += geom_line(size=0.6, alpha=0.4, color='#333333') 
        p_comb += geom_point(aes(color='Frame'), size=3, alpha=1, stroke=0)
        p_comb += scale_color_manual(values=frame_palette)
    else:
        # Monocolor style
        p_comb += geom_line(size=0.6, alpha=0.4, color='#333333')

    p_comb += labs(x=x_label_str, y='log2(Mean attention + 1)') 
    p_comb += theme_classic()
    p_comb += theme(axis_text_x=x_axis_text, axis_ticks_major_x=x_axis_ticks, axis_title_x=x_axis_title, figure_size=(weight, height))
    
    p_comb.save(f"{base_out}.combined.pdf")
    print(f"Combined attention profile saved to {base_out}.combined.pdf")

    # ==================================
    # Per-layer Plot
    # ==================================
    n_layers = df_plot['layer'].nunique()
    df_plot['Layer'] = pd.Categorical([f'L{li}' for li in df_plot['layer']], categories=[f'L{i}' for i in range(n_layers)])

    rect_per_layer = pd.DataFrame({
        'Layer': pd.Categorical([f'L{i}' for i in range(n_layers)], categories=[f'L{i}' for i in range(n_layers)]),
        'xmin': [0] * n_layers, 'xmax': [FIXED_CDS_LEN] * n_layers,
        'ymin': [-float('inf')] * n_layers, 'ymax': [float('inf')] * n_layers,
        'fill': ['lightgray'] * n_layers
    })

    p_layers = (
        ggplot(df_plot, aes(x='x_pos', y='log2_mean_attn'))
        + scale_fill_identity()
    )

    if show_cds:
        p_layers += geom_rect(data=rect_per_layer, mapping=aes(xmin='xmin', xmax='xmax', ymin='ymin', ymax='ymax', fill='fill'), alpha=0.3, inherit_aes=False, show_legend=False)
    
    if color_by_frame:
        p_layers += geom_line(size=0.4, alpha=0.4, color='#333333')
        p_layers += geom_point(aes(color='Frame'), size=2, alpha=1, stroke=0)
        p_layers += scale_color_manual(values=frame_palette)
    else:
        p_layers += geom_line(size=0.4, alpha=0.4, color='#333333')

    p_layers += facet_grid('Layer ~ .', scales='free_y')
    p_layers += labs(x=x_label_str, y='log2(Mean attention + 1)')
    p_layers += theme_classic()
    p_layers += theme(axis_text_x=x_axis_text, axis_ticks_major_x=x_axis_ticks, axis_title_x=x_axis_title, 
                      strip_background=element_blank(), strip_text=element_text(size=12), figure_size=(weight, height*3))
    
    p_layers.save(f"{base_out}.per_layer.pdf")
    print(f"Per-layer attention profile saved to {base_out}.per_layer.pdf")

    output_paths = [
        f"{base_out}.combined.pdf",
        f"{base_out}.per_layer.pdf",
    ]
    if has_head_profiles and not df_head_plot.empty:
        head_values = sorted(df_head_plot['head'].unique())
        head_labels = [f'H{head}' for head in head_values]
        layer_labels = [f'L{layer}' for layer in range(n_layers)]
        df_head_plot['Layer'] = pd.Categorical(
            [f'L{layer}' for layer in df_head_plot['layer']],
            categories=layer_labels,
        )
        df_head_plot['Head'] = pd.Categorical(
            [f'H{head}' for head in df_head_plot['head']],
            categories=head_labels,
        )

        per_head_group_columns = ['head', 'x_pos']
        if color_by_frame:
            per_head_group_columns.append('Frame')
        per_head_df = df_head_plot.groupby(
            per_head_group_columns, as_index=False, observed=True
        )[['mean_attn']].mean()
        per_head_df['log2_mean_attn'] = np.log2(
            per_head_df['mean_attn'] + 1
        )
        per_head_df['Head'] = pd.Categorical(
            [f'H{head}' for head in per_head_df['head']],
            categories=head_labels,
        )
        rect_standalone_head = pd.DataFrame({
            'Head': pd.Categorical(head_labels, categories=head_labels),
            'xmin': [0] * len(head_labels),
            'xmax': [FIXED_CDS_LEN] * len(head_labels),
            'ymin': [-float('inf')] * len(head_labels),
            'ymax': [float('inf')] * len(head_labels),
            'fill': ['lightgray'] * len(head_labels),
        })

        per_head_plot = (
            ggplot(per_head_df, aes(x='x_pos', y='log2_mean_attn'))
            + scale_fill_identity()
        )
        if show_cds:
            per_head_plot += geom_rect(
                data=rect_standalone_head,
                mapping=aes(
                    xmin='xmin', xmax='xmax', ymin='ymin', ymax='ymax',
                    fill='fill',
                ),
                alpha=0.3,
                inherit_aes=False,
                show_legend=False,
            )
        per_head_plot += geom_line(
            size=0.4, alpha=0.4, color='#333333'
        )
        if color_by_frame:
            per_head_plot += geom_point(
                aes(color='Frame'), size=1.5, alpha=1, stroke=0
            )
            per_head_plot += scale_color_manual(values=frame_palette)
        per_head_plot += facet_grid('Head ~ .', scales='free_y')
        per_head_plot += labs(
            x=x_label_str,
            y='log2(Mean attention across layers + 1)',
        )
        per_head_plot += theme_classic()
        per_head_plot += theme(
            axis_text_x=x_axis_text,
            axis_ticks_major_x=x_axis_ticks,
            axis_title_x=x_axis_title,
            strip_background=element_blank(),
            strip_text=element_text(size=10),
            figure_size=(
                weight,
                max(height, 0.7 * len(head_labels) + 2),
            ),
        )
        standalone_head_path = f"{base_out}.per_head.pdf"
        per_head_plot.save(standalone_head_path)
        print(f"Per-head attention profile saved to {standalone_head_path}")
        output_paths.append(standalone_head_path)

        layer_head_labels = [
            f'L{layer}-H{head}'
            for layer in range(n_layers)
            for head in head_values
        ]
        df_head_plot['Layer_Head'] = pd.Categorical(
            [
                f'L{layer}-H{head}'
                for layer, head in zip(
                    df_head_plot['layer'], df_head_plot['head']
                )
            ],
            categories=layer_head_labels,
        )
        rect_layer_head = pd.DataFrame({
            'Layer_Head': pd.Categorical(
                layer_head_labels,
                categories=layer_head_labels,
            ),
            'xmin': [0] * len(layer_head_labels),
            'xmax': [FIXED_CDS_LEN] * len(layer_head_labels),
            'ymin': [-float('inf')] * len(layer_head_labels),
            'ymax': [float('inf')] * len(layer_head_labels),
            'fill': ['lightgray'] * len(layer_head_labels),
        })
        layer_head_plot = (
            ggplot(df_head_plot, aes(x='x_pos', y='log2_mean_attn'))
            + scale_fill_identity()
        )
        if show_cds:
            layer_head_plot += geom_rect(
                data=rect_layer_head,
                mapping=aes(
                    xmin='xmin', xmax='xmax', ymin='ymin', ymax='ymax',
                    fill='fill',
                ),
                alpha=0.3,
                inherit_aes=False,
                show_legend=False,
            )
        layer_head_plot += geom_line(
            size=0.4, alpha=0.4, color='#333333'
        )
        if color_by_frame:
            layer_head_plot += geom_point(
                aes(color='Frame'), size=1.5, alpha=1, stroke=0
            )
            layer_head_plot += scale_color_manual(values=frame_palette)
        layer_head_plot += facet_grid('Layer_Head ~ .', scales='free_y')
        layer_head_plot += labs(
            x=x_label_str,
            y='log2(Mean attention + 1)',
        )
        layer_head_plot += theme_classic()
        layer_head_plot += theme(
            axis_text_x=x_axis_text,
            axis_ticks_major_x=x_axis_ticks,
            axis_title_x=x_axis_title,
            strip_background=element_blank(),
            strip_text=element_text(size=9),
            figure_size=(
                weight,
                max(height, 0.65 * len(layer_head_labels) + 2),
            ),
        )
        layer_head_path = f"{base_out}.per_layer_head.pdf"
        layer_head_plot.save(layer_head_path)
        print(
            "Per-layer and per-head attention profile saved to "
            f"{layer_head_path}"
        )
        output_paths.append(layer_head_path)

    elif not has_head_profiles:
        print(
            "Head-specific attention was not plotted because the input table "
            "has no 'head' column. Re-run "
            "extract_attention_positional_importance with the updated code."
        )

    return output_paths


def _prepare_attention_heatmap_matrix(
        attn_df,
        row_columns,
        up_len=300,
        down_len=300,
        xlim=None,
        position_bin_size=10,
        normalization='row_zscore'):
    """Aggregate an attention table into a row-by-position heatmap matrix."""
    if position_bin_size < 1:
        raise ValueError("position_bin_size must be a positive integer.")
    if xlim is not None:
        if len(xlim) != 2 or xlim[0] > xlim[1]:
            raise ValueError("xlim must contain two ordered coordinates.")
        lower_bound, upper_bound = xlim
    else:
        lower_bound = -up_len
        upper_bound = FIXED_CDS_LEN + down_len - 1

    required_columns = {'x_pos', 'mean_attn', *row_columns}
    missing_columns = required_columns.difference(attn_df.columns)
    if missing_columns:
        raise ValueError(
            f"Attention table is missing columns: {sorted(missing_columns)}"
        )
    working_df = attn_df[
        attn_df['x_pos'].between(lower_bound, upper_bound)
    ].copy()
    if working_df.empty:
        raise ValueError("No attention positions remain within the plot range.")

    working_df['position_bin'] = (
        np.floor(working_df['x_pos'] / position_bin_size).astype(int)
        * position_bin_size
    )
    grouped_df = working_df.groupby(
        [*row_columns, 'position_bin'], as_index=False, observed=True
    )['mean_attn'].mean()
    raw_matrix = grouped_df.pivot_table(
        index=row_columns,
        columns='position_bin',
        values='mean_attn',
        aggfunc='mean',
    ).sort_index(axis=1)
    number_of_bins_before = raw_matrix.shape[1]
    raw_matrix = raw_matrix.dropna(axis=1, how='any')
    number_of_bins_dropped = number_of_bins_before - raw_matrix.shape[1]
    if number_of_bins_dropped:
        print(
            f"[Attention heatmap] Dropped {number_of_bins_dropped} of "
            f"{number_of_bins_before} position bins because at least one "
            "row lacked a measurement."
        )
    if raw_matrix.empty:
        raise ValueError(
            "No position bins have complete attention measurements across "
            "all requested rows. Increase n_samples or position_bin_size."
        )

    valid_normalizations = {'none', 'row_fraction', 'row_zscore', 'row_minmax'}
    normalization = str(normalization).lower()
    if normalization not in valid_normalizations:
        raise ValueError(
            f"normalization must be one of {sorted(valid_normalizations)}."
        )
    if normalization == 'none':
        display_matrix = raw_matrix.copy()
    elif normalization == 'row_fraction':
        row_sums = raw_matrix.sum(axis=1).replace(0, np.nan)
        display_matrix = raw_matrix.div(row_sums, axis=0).fillna(0.0)
    elif normalization == 'row_minmax':
        row_min = raw_matrix.min(axis=1)
        row_range = (raw_matrix.max(axis=1) - row_min).replace(0, np.nan)
        display_matrix = raw_matrix.sub(row_min, axis=0).div(
            row_range, axis=0
        ).fillna(0.0)
    else:
        row_mean = raw_matrix.mean(axis=1)
        row_std = raw_matrix.std(axis=1, ddof=0).replace(0, np.nan)
        display_matrix = raw_matrix.sub(row_mean, axis=0).div(
            row_std, axis=0
        ).fillna(0.0)

    return raw_matrix, display_matrix


def _infer_attention_focus(raw_matrix, enrichment_threshold=1.15):
    """Classify each attention row by its dominant transcript region."""
    if enrichment_threshold <= 1:
        raise ValueError("enrichment_threshold must be greater than 1.")
    positions = np.asarray(raw_matrix.columns, dtype=float)
    region_masks = {
        "5' UTR": positions < 0,
        'CDS': (positions >= 0) & (positions < FIXED_CDS_LEN),
        "3' UTR": positions >= FIXED_CDS_LEN,
    }
    available_regions = [
        region for region, mask in region_masks.items() if mask.any()
    ]
    focus_labels = {}
    for row_label, row_values in raw_matrix.iterrows():
        region_means = {
            region: float(row_values.iloc[np.flatnonzero(region_masks[region])].mean())
            for region in available_regions
        }
        if len(region_means) < 2 or not any(
                np.isfinite(list(region_means.values()))):
            focus_labels[row_label] = 'Full length'
            continue
        best_region = max(region_means, key=region_means.get)
        other_values = [
            value for region, value in region_means.items()
            if region != best_region
        ]
        reference = float(np.mean(other_values))
        enrichment = (
            (region_means[best_region] + 1e-12) / (reference + 1e-12)
        )
        focus_labels[row_label] = (
            best_region if enrichment >= enrichment_threshold
            else 'Full length'
        )
    return pd.Series(focus_labels, name='Attention focus')


def plot_attention_profile_heatmap(
        attn_df,
        out_path="attention_profile_heatmap.pdf",
        up_len=300,
        down_len=300,
        xlim=None,
        position_bin_size=10,
        normalization='row_zscore',
        cluster_layers=True,
        cluster_heads=True,
        cluster_metric='correlation',
        cluster_method='average',
        head_mode='head',
        enrichment_threshold=1.15,
        cmap=None,
        vmin=None,
        vmax=None,
        layer_width=12,
        layer_height=6,
        head_width=12,
        head_height=7,
        font_size=8,
        show_region_colors=True,
        show_focus_colors=True):
    """Plot clustered layer and head attention-position heatmaps.

    Rows are clustered while metagene columns retain their biological 5'-to-3'
    order. ``head_mode='head'`` averages the same head index across layers;
    ``head_mode='layer_head'`` treats every layer-head pair independently.
    """
    import seaborn as sns
    from matplotlib.patches import Patch
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist

    if 'head' not in attn_df.columns:
        raise ValueError(
            "The attention table has no 'head' column. Re-run "
            "extract_attention_positional_importance with the updated code."
        )
    head_mode = str(head_mode).lower()
    normalization = str(normalization).lower()
    cluster_metric = str(cluster_metric).lower()
    cluster_method = str(cluster_method).lower()
    if head_mode not in {'head', 'layer_head'}:
        raise ValueError("head_mode must be 'head' or 'layer_head'.")
    for dimension_name, dimension_value in {
            'layer_width': layer_width,
            'layer_height': layer_height,
            'head_width': head_width,
            'head_height': head_height,
    }.items():
        if dimension_value <= 0:
            raise ValueError(f"{dimension_name} must be positive.")

    layer_raw, layer_matrix = _prepare_attention_heatmap_matrix(
        attn_df=attn_df,
        row_columns=['layer'],
        up_len=up_len,
        down_len=down_len,
        xlim=xlim,
        position_bin_size=position_bin_size,
        normalization=normalization,
    )
    head_row_columns = ['head'] if head_mode == 'head' else ['layer', 'head']
    head_raw, head_matrix = _prepare_attention_heatmap_matrix(
        attn_df=attn_df,
        row_columns=head_row_columns,
        up_len=up_len,
        down_len=down_len,
        xlim=xlim,
        position_bin_size=position_bin_size,
        normalization=normalization,
    )

    layer_raw.index = [f'L{int(value)}' for value in layer_raw.index]
    layer_matrix.index = layer_raw.index
    if head_mode == 'head':
        head_raw.index = [f'H{int(value)}' for value in head_raw.index]
    else:
        head_raw.index = [
            f'L{int(layer)}-H{int(head)}'
            for layer, head in head_raw.index
        ]
    head_matrix.index = head_raw.index

    focus_palette = {
        "5' UTR": '#E69F00',
        'CDS': '#D55E00',
        "3' UTR": '#7B61A8',
        'Full length': '#8A8A8A',
    }
    region_palette = {
        "5' UTR": '#E69F00',
        'CDS': '#56B4E9',
        "3' UTR": '#7B61A8',
    }

    def build_linkage(matrix, enabled):
        if not enabled or len(matrix) < 2:
            return None
        if cluster_method == 'ward' and cluster_metric != 'euclidean':
            raise ValueError(
                "Ward linkage requires cluster_metric='euclidean'."
            )
        distances = pdist(matrix.to_numpy(dtype=float), metric=cluster_metric)
        distances = np.nan_to_num(
            distances, nan=1.0, posinf=1.0, neginf=0.0
        )
        if not np.any(distances > 0):
            return None
        return linkage(
            distances,
            method=cluster_method,
            optimal_ordering=True,
        )

    def draw_heatmap(
            raw_matrix,
            display_matrix,
            cluster_rows,
            width,
            height,
            title,
            output_path):
        focus = _infer_attention_focus(
            raw_matrix,
            enrichment_threshold=enrichment_threshold,
        )
        row_colors = (
            focus.map(focus_palette) if show_focus_colors else None
        )
        position_values = np.asarray(display_matrix.columns, dtype=float)
        position_regions = pd.Series(
            np.select(
                [
                    position_values < 0,
                    (position_values >= 0)
                    & (position_values < FIXED_CDS_LEN),
                    position_values >= FIXED_CDS_LEN,
                ],
                ["5' UTR", 'CDS', "3' UTR"],
                default='CDS',
            ),
            index=display_matrix.columns,
            name='Transcript region',
        )
        column_colors = (
            position_regions.map(region_palette)
            if show_region_colors else None
        )
        row_linkage = build_linkage(display_matrix, cluster_rows)
        use_row_clustering = row_linkage is not None
        selected_cmap = cmap or (
            'vlag' if normalization == 'row_zscore' else 'mako'
        )
        center = 0 if normalization == 'row_zscore' else None
        grid = sns.clustermap(
            display_matrix,
            row_cluster=use_row_clustering,
            row_linkage=row_linkage,
            col_cluster=False,
            row_colors=row_colors,
            col_colors=column_colors,
            cmap=selected_cmap,
            center=center,
            vmin=vmin,
            vmax=vmax,
            xticklabels=False,
            yticklabels=True,
            linewidths=0,
            figsize=(width, height),
            dendrogram_ratio=(0.14, 0.05),
            colors_ratio=(0.025, 0.025),
            cbar_pos=(0.02, 0.80, 0.02, 0.15),
            cbar_kws={'label': normalization.replace('_', ' ').title()},
        )
        grid.ax_heatmap.set_title(title, fontsize=font_size + 2, pad=10)
        grid.ax_heatmap.set_xlabel('Metagene position', fontsize=font_size)
        grid.ax_heatmap.set_ylabel('')
        grid.ax_heatmap.tick_params(axis='y', labelsize=font_size)

        number_of_ticks = min(9, len(display_matrix.columns))
        tick_indices = np.linspace(
            0, len(display_matrix.columns) - 1, number_of_ticks
        ).astype(int)
        grid.ax_heatmap.set_xticks(tick_indices + 0.5)
        grid.ax_heatmap.set_xticklabels(
            [
                str(int(display_matrix.columns[index]))
                for index in tick_indices
            ],
            rotation=0,
            fontsize=font_size,
        )

        legend_handles = []
        if show_focus_colors:
            legend_handles.extend([
                Patch(color=color, label=label)
                for label, color in focus_palette.items()
            ])
        if show_region_colors:
            legend_handles.extend([
                Patch(
                    facecolor=color,
                    edgecolor='none',
                    label=f'Position: {label}',
                )
                for label, color in region_palette.items()
            ])
        if legend_handles:
            grid.ax_heatmap.legend(
                handles=legend_handles,
                title='Annotations',
                frameon=False,
                fontsize=font_size,
                title_fontsize=font_size,
                bbox_to_anchor=(1.02, 1),
                loc='upper left',
                borderaxespad=0,
            )
        grid.fig.savefig(output_path, bbox_inches='tight')
        row_order = (
            grid.dendrogram_row.reordered_ind
            if use_row_clustering else list(range(len(display_matrix)))
        )
        ordered_labels = display_matrix.index[row_order].tolist()
        plt.close(grid.fig)
        return ordered_labels, focus

    base_out = os.path.splitext(os.fspath(out_path))[0]
    layer_path = f"{base_out}.layers.pdf"
    head_path = f"{base_out}.heads.pdf"
    layer_order, layer_focus = draw_heatmap(
        raw_matrix=layer_raw,
        display_matrix=layer_matrix,
        cluster_rows=cluster_layers,
        width=layer_width,
        height=layer_height,
        title='Layer attention profiles',
        output_path=layer_path,
    )
    head_title = (
        'Head attention profiles (averaged across layers)'
        if head_mode == 'head'
        else 'Layer-head attention profiles'
    )
    head_order, head_focus = draw_heatmap(
        raw_matrix=head_raw,
        display_matrix=head_matrix,
        cluster_rows=cluster_heads,
        width=head_width,
        height=head_height,
        title=head_title,
        output_path=head_path,
    )
    print(f"Layer attention heatmap saved to {layer_path}")
    print(f"Head attention heatmap saved to {head_path}")
    return {
        'paths': [layer_path, head_path],
        'layer_matrix': layer_matrix,
        'head_matrix': head_matrix,
        'layer_order': layer_order,
        'head_order': head_order,
        'layer_focus': layer_focus,
        'head_focus': head_focus,
    }


def plot_regional_attention_dynamics(attn_df, out_path="regional_attention_dynamics.pdf", up_len=300, down_len=300):
    """
    Plots the layer-by-layer dynamic shifts in attention across 5 specific regions:
    5' UTR, CDS (Frame 0), CDS (Frame 1), CDS (Frame 2), and 3' UTR.
    Produces both an absolute mean attention line plot and a 100% relative proportion bar chart.
    """
    from plotnine import (ggplot, aes, geom_line, geom_point, geom_col, position_stack,
                          labs, theme_classic, scale_color_manual, scale_fill_manual, 
                          scale_x_continuous, theme, element_text)
    # 1. Filter sequences based on the upstream/downstream boundaries
    df = attn_df[(attn_df['x_pos'] >= -up_len) & (attn_df['x_pos'] <= FIXED_CDS_LEN + down_len - 1)].copy()

    # 2. Annotate the 5 regions
    conditions = [
        df['x_pos'] < 0,
        (df['x_pos'] >= 0) & (df['x_pos'] < FIXED_CDS_LEN) & (df['x_pos'] % 3 == 0),
        (df['x_pos'] >= 0) & (df['x_pos'] < FIXED_CDS_LEN) & (df['x_pos'] % 3 == 1),
        (df['x_pos'] >= 0) & (df['x_pos'] < FIXED_CDS_LEN) & (df['x_pos'] % 3 == 2),
        df['x_pos'] >= FIXED_CDS_LEN
    ]
    choices = ["5' UTR", "CDS (Frame 0)", "CDS (Frame 1)", "CDS (Frame 2)", "3' UTR"]
    
    df['Region'] = np.select(conditions, choices, default="Unknown")

    # Convert to Categorical to maintain a strict legend order
    region_order = ["5' UTR", "CDS (Frame 0)", "CDS (Frame 1)", "CDS (Frame 2)", "3' UTR"]
    df['Region'] = pd.Categorical(df['Region'], categories=region_order)

    # 3. Aggregate Mean Attention per Region per Layer
    # Using 'mean' perfectly balances the varying lengths of UTRs and CDS subsets
    agg_df = df.groupby(['layer', 'Region'], as_index=False, observed=True)[['mean_attn']].mean()
    
    # Scale up by 1000 for cleaner Y-axis numbers in the absolute plot
    agg_df['mean_attn_scaled'] = agg_df['mean_attn']

    # Define color map to perfectly match your previous plots
    color_map = {
        "5' UTR": "#FF7F00",       # Orange for 5' UTR
        "CDS (Frame 0)": "#E41A1C", # Red
        "CDS (Frame 1)": "#377EB8", # Blue
        "CDS (Frame 2)": "gray",    # Gray
        "3' UTR": "#984EA3"         # Purple for 3' UTR
    }

    base_out = os.path.splitext(os.fspath(out_path))[0]
    max_layer = int(agg_df['layer'].max())

    # ============================================================
    # Plot 1: Absolute Mean Attention per Nucleotide (Line Plot)
    # ============================================================
    p_line = (
        ggplot(agg_df, aes(x='layer', y='mean_attn_scaled', color='Region', group='Region'))
        + geom_line(size=1.2, alpha=0.9)
        + geom_point(size=3)
        + scale_color_manual(values=color_map)
        + scale_x_continuous(breaks=range(0, max_layer + 1))
        + labs(x='Transformer Layer', 
               y='Mean Attention per nt', 
               title='Layer-wise Absolute Attention Dynamics')
        + theme_classic()
        + theme(figure_size=(7, 5),
                axis_text=element_text(size=10),
                title=element_text(size=12, face="bold"))
    )
    p_line.save(f"{base_out}.line.pdf")
    print(f"Regional dynamics (Line) saved to {base_out}.line.pdf")

    # ============================================================
    # Plot 2: Relative Contribution Proportion (100% Stacked Bar)
    # ============================================================
    # Calculate relative proportion for each layer
    layer_sums = agg_df.groupby('layer')['mean_attn'].transform('sum')
    agg_df['relative_prop'] = agg_df['mean_attn'] / layer_sums

    p_bar = (
        ggplot(agg_df, aes(x='layer', y='relative_prop', fill='Region'))
        # Reverse the stack so 5'UTR is at the bottom, matching 5'->3' direction intuitively
        + geom_col(position=position_stack(reverse=True), color='white', size=0.2)
        + scale_fill_manual(values=color_map)
        + scale_x_continuous(breaks=range(0, max_layer + 1))
        + labs(x='Transformer Layer', 
               y='Relative Regional Contribution (100%)', 
               title='Layer-wise Relative Attention Shift')
        + theme_classic()
        + theme(figure_size=(7, 5),
                axis_text=element_text(size=10),
                title=element_text(size=12, face="bold"))
    )
    p_bar.save(f"{base_out}.proportion.pdf")
    print(f"Regional dynamics (Proportion Bar) saved to {base_out}.proportion.pdf")


def plot_saliency_profile(
        sal_df,
        out_path="saliency_profile.pdf",
        up_len=300,
        down_len=300,
        color_by_frame=True,
        xlim=None,
        show_xaxis=False,
        show_cds=True,
        weight=6,
        height=5):
    """Plot a saliency profile with attention-profile-compatible controls."""
    from plotnine import (
        ggplot, aes, geom_point, geom_line, geom_rect, labs, theme_classic,
        theme, scale_color_manual, scale_fill_identity, element_blank,
        element_line, element_text,
    )

    required_columns = {'x_pos', 'mean_saliency'}
    missing_columns = required_columns.difference(sal_df.columns)
    if missing_columns:
        raise ValueError(
            f"Saliency table is missing columns: {sorted(missing_columns)}"
        )

    if xlim is not None:
        if len(xlim) != 2 or xlim[0] > xlim[1]:
            raise ValueError("xlim must contain two ordered coordinates.")
        lower_bound, upper_bound = xlim
    else:
        lower_bound = -up_len
        upper_bound = FIXED_CDS_LEN + down_len - 1
    df_plot = sal_df[
        sal_df['x_pos'].between(lower_bound, upper_bound)
    ].copy()
    if df_plot.empty:
        raise ValueError("No saliency positions remain within the plot range.")

    group_columns = ['x_pos']
    if color_by_frame:
        df_plot = _assign_frame_colors(df_plot)
        group_columns.append('Frame')
    df_plot = df_plot.groupby(
        group_columns, as_index=False, observed=True
    )[['mean_saliency']].mean().dropna(subset=['mean_saliency'])
    df_plot['log2_saliency'] = np.log2(df_plot['mean_saliency'] + 1)

    frame_palette = {
        'Frame 0': '#E41A1C',
        'Frame 1': '#377EB8',
        'Frame 2': 'gray',
    }
    rect_cds = _cds_rect_data()
    x_axis_text = element_text() if show_xaxis else element_blank()
    x_axis_ticks = element_line() if show_xaxis else element_blank()
    x_axis_title = element_text() if show_xaxis else element_blank()
    x_label = 'Metagene Position (x_pos)' if show_xaxis else ''

    plot = (
        ggplot(df_plot, aes(x='x_pos', y='log2_saliency'))
        + scale_fill_identity()
    )
    if show_cds:
        plot += geom_rect(
            data=rect_cds,
            mapping=aes(
                xmin='xmin', xmax='xmax', ymin='ymin', ymax='ymax',
                fill='fill',
            ),
            alpha=0.3,
            inherit_aes=False,
            show_legend=False,
        )
    plot += geom_line(size=0.6, alpha=0.4, color='#333333')
    if color_by_frame:
        plot += geom_point(
            aes(color='Frame'), size=2, alpha=1, stroke=0
        )
        plot += scale_color_manual(values=frame_palette)
    plot += labs(
        x=x_label,
        y='log2(Mean |d(profile)/d(base)| + 1)',
    )
    plot += theme_classic()
    plot += theme(
        axis_text_x=x_axis_text,
        axis_ticks_major_x=x_axis_ticks,
        axis_title_x=x_axis_title,
        figure_size=(weight, height),
    )

    requested_path = os.fspath(out_path)
    base_path, extension = os.path.splitext(requested_path)
    pdf_path = (
        requested_path if extension.lower() == '.pdf'
        else f"{base_path or requested_path}.pdf"
    )
    plot.save(pdf_path)
    print(f"Saliency profile saved to {pdf_path}")
    return pdf_path


def plot_sequence_logo(
        sequences,
        title="Sequence Motif",
        out_path="sequence_logo.pdf"):
    """Plot an information-content sequence logo and save it as PDF."""
    if not sequences:
        print(f"No sequences found for {title}.")
        return None
        
    # Convert aligned sequences to an information-content matrix.
    df = logomaker.alignment_to_matrix(sequences=sequences, to_type='information')
    
    fig, ax = plt.subplots(figsize=(8, 3))
    logo = logomaker.Logo(df, ax=ax, font_name='Arial Rounded MT Bold')
    logo.style_spines(visible=False)
    logo.style_spines(spines=['left', 'bottom'], visible=True)
    ax.set_ylabel('Information (bits)')
    ax.set_xlabel('Relative Position')
    plt.title(title)
    pdf_path = _as_pdf_path(out_path)
    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Sequence logo saved to {pdf_path}")
    return pdf_path


def plot_cluster_sequence_logos(
        clustered_df,
        region_name,
        out_dir,
        minimum_support=5,
        width=7,
        height=2.2):
    """Plot one sequence logo per motif cluster and return PDF paths."""
    required_columns = {'Cluster_ID', 'sequence'}
    missing_columns = required_columns.difference(clustered_df.columns)
    if missing_columns:
        raise ValueError(
            f"Cluster table is missing columns: {sorted(missing_columns)}"
        )
    os.makedirs(out_dir, exist_ok=True)
    output_paths = []
    for cluster_id in sorted(clustered_df['Cluster_ID'].dropna().unique()):
        sequences = clustered_df.loc[
            clustered_df['Cluster_ID'] == cluster_id,
            'sequence',
        ].tolist()
        if len(sequences) < minimum_support:
            continue
        information_matrix = logomaker.alignment_to_matrix(
            sequences=sequences,
            to_type='information',
        )
        fig, ax = plt.subplots(figsize=(width, height))
        logo = logomaker.Logo(information_matrix, ax=ax)
        logo.style_spines(visible=False)
        logo.style_spines(spines=['left', 'bottom'], visible=True)
        ax.set_ylabel('Information (bits)')
        ax.set_xlabel('Relative Offset from Peak')
        ax.set_title(
            f"{region_name} - Cluster {cluster_id} "
            f"(Support: n={len(sequences)})"
        )
        output_path = os.path.join(
            out_dir,
            f"motif_logo_{region_name}_cluster_{cluster_id}.pdf",
        )
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        output_paths.append(output_path)
    return output_paths


def plot_motif_metagene_heatmap(
        all_motifs_df, out_path="motif_metagene_heatmap.pdf", 
        bin_size=20, up_len=300, down_len=300, max_prob=None,
        weight=8, height=10
        ):
    """
    Plots a heatmap of motif spatial distribution along the metagene.
    Y-axis motifs are dynamically sorted from 5' to 3' based on their peak enrichment positions.
    """
    from plotnine import (ggplot, aes, geom_tile, geom_vline, scale_fill_gradient, 
                          labs, theme_classic, theme, element_text, element_blank, element_line)
    
    if 'Motif_Name' not in all_motifs_df.columns or all_motifs_df.empty:
        print("No motif clustering data available to plot.")
        return

    # 1. Filter coordinate boundaries
    df_plot = all_motifs_df[(all_motifs_df['x_pos'] >= -up_len) & (all_motifs_df['x_pos'] <= FIXED_CDS_LEN + down_len)].copy()
    
    # 2. Perform positional binning
    df_plot['x_bin'] = (df_plot['x_pos'] // bin_size) * bin_size + (bin_size / 2)
    
    # 3. Calculate raw frequencies
    heatmap_data = df_plot.groupby(['Motif_Name', 'x_bin']).size().reset_index(name='count')
    
    # 4. Impute empty grid tiles to guarantee a continuous matrix background
    unique_motifs = heatmap_data['Motif_Name'].unique()
    min_bin = (-up_len // bin_size) * bin_size + (bin_size / 2)
    max_bin = ((FIXED_CDS_LEN + down_len) // bin_size) * bin_size + (bin_size / 2)
    all_bins = np.arange(min_bin, max_bin + bin_size, bin_size)
    
    full_index = pd.MultiIndex.from_product([unique_motifs, all_bins], names=['Motif_Name', 'x_bin'])
    full_df = pd.DataFrame(index=full_index).reset_index()
    
    heatmap_data = pd.merge(full_df, heatmap_data, on=['Motif_Name', 'x_bin'], how='left').fillna({'count': 0})
    
    # 5. Standardize via row-wise probability
    motif_totals = heatmap_data.groupby('Motif_Name')['count'].transform('sum')
    heatmap_data['Probability'] = heatmap_data['count'] / (motif_totals + 1e-9)

    # Sort motifs by their strongest positional enrichment from 5' to 3'.
    peak_bins = heatmap_data.loc[heatmap_data.groupby('Motif_Name')['Probability'].idxmax()]
    
    # Use motif name as a stable secondary key for tied peak positions.
    ordered_motifs = peak_bins.sort_values(
        ['x_bin', 'Motif_Name'], 
        ascending=[False, False]
    )['Motif_Name'].tolist()

    # Apply the resulting categorical order.
    heatmap_data['Motif_Name'] = pd.Categorical(
        heatmap_data['Motif_Name'], 
        categories=ordered_motifs
    )
    
    # 7. Automatically upscale the color scale if the global probability is too low
    if max_prob is None:
        max_prob = heatmap_data['Probability'].max()
        if max_prob == 0:
            max_prob = 1.0
    
    # 8. Render plot configurations
    p = (
        ggplot(heatmap_data, aes(x='x_bin', y='Motif_Name', fill='Probability'))
        + geom_tile(color='white', size=0.1) 
        + scale_fill_gradient(low='#EFF3FF', high='#08306B', limits=(0, max_prob)) 
        + geom_vline(xintercept=[0, FIXED_CDS_LEN], linetype='dashed', color='red', size=0.6)
        + labs(
            x=f'Metagene Position (Bin Size = {bin_size} nt)', 
            y='Discovered Motifs (Sorted 5\' \u2192 3\')', 
            fill='Spatial\nProbability', 
            title='Motif Spatial Distribution along Metagene'
        )
        + theme_classic()
        + theme(
            figure_size=(weight, height),
            axis_text_y=element_text(size=10),
            axis_text_x=element_text(size=10),
            axis_ticks_major_x=element_line(color='#333333', size=0.5),
            axis_ticks_major_y=element_line(color='#333333', size=0.5),
            axis_line_x=element_blank(),
            axis_line_y=element_blank()
        )
    )

    pdf_path = _as_pdf_path(out_path)
    p.save(pdf_path)
    print(f"Motif metagene heatmap saved to {pdf_path}")
    return pdf_path
