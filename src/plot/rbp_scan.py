"""Plots for RBP motif scanning results."""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})


def _as_pdf_path(path):
    """Return an output path with a PDF suffix."""
    return f"{os.path.splitext(str(path))[0]}.pdf"

def plot_rbp_metagene_heatmap(mapped_peaks_df, out_path, FIXED_CDS_LEN=600, bin_size=20, up_len=300, down_len=300):
    """Plot a metagene probability heatmap of RBP binding sites."""
    from plotnine import (
        aes, element_blank, element_text, geom_tile, geom_vline, ggplot,
        labs, scale_fill_gradient, theme, theme_classic,
    )

    if mapped_peaks_df.empty: return
    
    df_plot = mapped_peaks_df[(mapped_peaks_df['x_pos'] >= -up_len) & (mapped_peaks_df['x_pos'] <= FIXED_CDS_LEN + down_len)].copy()
    df_plot['x_bin'] = (df_plot['x_pos'] // bin_size) * bin_size + (bin_size / 2)
    
    heatmap_data = df_plot.groupby(['RBP_Name', 'x_bin']).size().reset_index(name='count')
    
    # Fill the complete RBP-by-position grid.
    unique_rbps = heatmap_data['RBP_Name'].unique()
    all_bins = np.arange((-up_len // bin_size) * bin_size + (bin_size / 2), 
                         ((FIXED_CDS_LEN + down_len) // bin_size) * bin_size + (bin_size / 2) + bin_size, 
                         bin_size)
    full_df = pd.DataFrame(index=pd.MultiIndex.from_product([unique_rbps, all_bins], names=['RBP_Name', 'x_bin'])).reset_index()
    heatmap_data = pd.merge(full_df, heatmap_data, on=['RBP_Name', 'x_bin'], how='left').fillna({'count': 0})
    
    # Normalize each row to represent positional preference.
    rbp_totals = heatmap_data.groupby('RBP_Name')['count'].transform('sum')
    heatmap_data['Probability'] = heatmap_data['count'] / (rbp_totals + 1e-9)
    
    # Order RBPs by their strongest position from 5' to 3'.
    peak_bins = heatmap_data.loc[heatmap_data.groupby('RBP_Name')['Probability'].idxmax()]
    ordered_rbps = peak_bins.sort_values(['x_bin', 'RBP_Name'], ascending=[False, False])['RBP_Name'].tolist()
    heatmap_data['RBP_Name'] = pd.Categorical(heatmap_data['RBP_Name'], categories=ordered_rbps)
    
    p = (
        ggplot(heatmap_data, aes(x='x_bin', y='RBP_Name', fill='Probability'))
        + geom_tile(color='white', size=0.1) 
        + scale_fill_gradient(low='#EFF3FF', high='#08306B', limits=(0, heatmap_data['Probability'].max() or 1.0)) 
        + geom_vline(xintercept=[0, FIXED_CDS_LEN], linetype='dashed', color='red', size=0.8)
        + labs(x=f'Metagene Position', y='RNA Binding Proteins', fill='Spatial\nProbability', title='RBP Spatial Distribution')
        + theme_classic()
        + theme(
            figure_size=(10, max(4, 0.15 * len(unique_rbps))),
            axis_text_y=element_text(size=9), 
            axis_line_x=element_blank(), 
            axis_line_y=element_blank()
            )
    )
    pdf_path = _as_pdf_path(out_path)
    p.save(pdf_path)
    print(f"RBP heatmap saved to {pdf_path}")
    return pdf_path


def plot_rbp_regulatory_bubble(rbp_landscape_df, out_path,
                                top_n_label=10, figsize=(9, 7)):
    """
    Bubble plot: RBP regulatory potential from attention and TE enrichment.

    X-axis: Normalized mean attention score (RBP-binding peaks with
            higher attention → model relies on that region more).
    Y-axis: Top-20% TE enrichment ratio
            (n_top_hit / n_bottom_hit, pseudocount +1).
    Bubble size: Total_Hits (log2-scaled).
    Color: 5'-to-3' preference (purple=5'UTR, green=CDS, orange=3'UTR).

    Top `top_n_label` RBPs in the upper-right quadrant are labeled.
    No motif logos are drawn — this is a clean quantitative overview.
    """
    df = rbp_landscape_df.dropna(subset=['Mean_Attention', 'Enrichment_Ratio']).copy()
    if df.empty:
        print("No RBPs with valid attention + enrichment data.")
        return

    # Color by dominant spatial preference.
    def dominant_region(row):
        regions = {'5UTR': row['5UTR_Hits'], 'CDS': row['CDS_Hits'], '3UTR': row['3UTR_Hits']}
        return max(regions, key=regions.get)

    region_colors = {'5UTR': '#7B3294', 'CDS': '#238B45', '3UTR': '#D95F02'}
    df['dominant'] = df.apply(dominant_region, axis=1)
    df['color'] = df['dominant'].map(region_colors)

    # Normalize attention to [0, 1] for interpretability.
    attn_raw = df['Mean_Attention'].values
    attn_norm = (attn_raw - attn_raw.min()) / (attn_raw.max() - attn_raw.min() + 1e-8)
    df['attn_norm'] = attn_norm

    sizes = np.log2(df['Total_Hits'].values + 1) * 18

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        df['attn_norm'], df['Enrichment_Ratio'],
        s=sizes, c=df['color'], alpha=0.7, edgecolors='#555555', linewidth=0.4,
    )

    # Add reference thresholds.
    ax.axhline(y=1.0, linestyle='--', color='#888888', linewidth=0.8, alpha=0.6)
    ax.axvline(x=0.5, linestyle='--', color='#888888', linewidth=0.8, alpha=0.6)

    # Label the strongest RBPs in the upper-right quadrant.
    upper_right = df[(df['attn_norm'] > 0.5) & (df['Enrichment_Ratio'] > 1.0)]
    upper_right = upper_right.nlargest(top_n_label, 'Total_Hits')

    for _, row in upper_right.iterrows():
        ax.annotate(
            row['RBP_Name'],
            (row['attn_norm'], row['Enrichment_Ratio']),
            textcoords="offset points", xytext=(15, 5),
            fontsize=7.5, fontweight='bold', alpha=0.85,
            arrowprops=dict(arrowstyle='->', color='#555555', lw=0.6),
        )

    # Build the region legend.
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=region_colors[r],
               markersize=9, label=f"{r} ({'5′UTR' if r == '5UTR' else r})")
        for r in ['5UTR', 'CDS', '3UTR']
    ]
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.85,
              fontsize=9, title='Dominant Region')

    # Style axes and labels.
    ax.set_xlabel("Normalized Mean Attention Score", fontsize=12)
    ax.set_ylabel("High / Low TE Enrichment Ratio", fontsize=12)
    ax.set_title("RBP Regulatory Landscape", fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    pdf_path = _as_pdf_path(out_path)
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"RBP regulatory bubble plot saved to {pdf_path}")
    return pdf_path


def plot_rbp_translation_effect_summary(
        summary_df,
        out_path="rbp_translation_effect_summary.pdf",
        top_n_per_direction=12,
        fdr_threshold=None,
        width=6.2,
        row_height=0.28):
    """Plot top positive and negative matched RBP-motif effects."""
    required = {
        'RBP_Name', 'Region', 'N_Transcripts', 'Median_Delta_Log2_TE',
        'CI_Lower', 'CI_Upper', 'FDR_BH', 'Direction',
    }
    missing = required.difference(summary_df.columns)
    if missing:
        raise ValueError(f"Summary table is missing columns: {sorted(missing)}")
    input_count = len(summary_df)
    working = summary_df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=['Median_Delta_Log2_TE', 'CI_Lower', 'CI_Upper']
    ).copy()
    nonfinite_removed = input_count - len(working)
    if fdr_threshold is not None:
        before_fdr = len(working)
        working = working[working['FDR_BH'] <= fdr_threshold]
        fdr_removed = before_fdr - len(working)
    else:
        fdr_removed = 0
    print(
        f"RBP summary rows: input={input_count}, "
        f"nonfinite_removed={nonfinite_removed}, "
        f"fdr_removed={fdr_removed}."
    )
    positive = working[working['Median_Delta_Log2_TE'] > 0].nlargest(
        top_n_per_direction, 'Median_Delta_Log2_TE'
    )
    negative = working[working['Median_Delta_Log2_TE'] < 0].nsmallest(
        top_n_per_direction, 'Median_Delta_Log2_TE'
    )
    plot_df = pd.concat([negative, positive], ignore_index=True).sort_values(
        'Median_Delta_Log2_TE'
    )
    if plot_df.empty:
        raise ValueError("No RBP effects remain after filtering.")

    labels = (
        plot_df['RBP_Name'].astype(str)
        + '  (' + plot_df['Region'].astype(str)
        + '; n=' + plot_df['N_Transcripts'].astype(int).astype(str) + ')'
    )
    positions = np.arange(len(plot_df))
    values = plot_df['Median_Delta_Log2_TE'].to_numpy(float)
    lower = values - plot_df['CI_Lower'].to_numpy(float)
    upper = plot_df['CI_Upper'].to_numpy(float) - values
    colors = np.where(values >= 0, '#C44E52', '#3B6FB6')
    sizes = 18 + 9 * np.sqrt(plot_df['N_Transcripts'].to_numpy(float))

    height = max(3.2, 1.5 + row_height * len(plot_df))
    fig, ax = plt.subplots(figsize=(width, height))
    for position, value, low, high, color in zip(
        positions, values, lower, upper, colors
    ):
        ax.errorbar(
            value, position, xerr=np.array([[low], [high]]),
            fmt='none', ecolor=color, elinewidth=1.1, capsize=2.2,
            alpha=0.9, zorder=1,
        )
    ax.scatter(
        values, positions, s=sizes, c=colors, edgecolor='white',
        linewidth=0.6, zorder=2,
    )
    ax.axvline(0, color='#555555', linewidth=0.8, linestyle='--')
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel(
        r'Motif contribution to predicted CDS translation, '
        r'$\Delta\log_2(TE)$'
    )
    ax.set_ylabel('')
    ax.grid(axis='x', color='#E6E6E6', linewidth=0.6)
    ax.set_axisbelow(True)
    ax.text(
        0.01, 1.015, 'Translation-suppressive motif',
        transform=ax.transAxes, ha='left', va='bottom', color='#3B6FB6',
        fontsize=8,
    )
    ax.text(
        0.99, 1.015, 'Translation-supportive motif',
        transform=ax.transAxes, ha='right', va='bottom', color='#C44E52',
        fontsize=8,
    )
    fig.tight_layout()
    pdf_path = _as_pdf_path(out_path)
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f"RBP translation-effect summary saved to {pdf_path}")
    return pdf_path


def _native_contribution_matrix(case_df):
    positions = case_df['Relative_Position'].astype(int).to_numpy()
    index = np.arange(positions.min(), positions.max() + 1)
    matrix = pd.DataFrame(0.0, index=index, columns=list('ACGT'))
    for row in case_df.itertuples(index=False):
        if row.Base in matrix.columns:
            matrix.loc[int(row.Relative_Position), row.Base] = float(
                row.Base_Contribution_Log2_TE
            )
    return matrix


def plot_rbp_nucleotide_contribution_cases(
        contribution_df,
        out_dir,
        max_cases=None,
        width=8.0,
        height=3.4,
        motif_color='#F3E7A6'):
    """Plot transcript context and signed per-base contributions for each case."""
    import logomaker

    required = {
        'Hit_ID', 'Tid', 'RBP_Name', 'Region', 'Motif_Start', 'Motif_End',
        'Absolute_Position', 'Relative_Position', 'Base', 'Is_Motif',
        'Base_Contribution_Log2_TE', 'Motif_Delta_Log2_TE',
        'Group_Median_Delta_Log2_TE', 'Group_N_Transcripts',
        'CDS_Start_0based', 'CDS_End_exclusive', 'Transcript_Length',
    }
    missing = required.difference(contribution_df.columns)
    if missing:
        raise ValueError(
            f"Contribution table is missing columns: {sorted(missing)}"
        )
    if contribution_df.empty:
        return []
    os.makedirs(out_dir, exist_ok=True)
    hit_ids = contribution_df['Hit_ID'].drop_duplicates().tolist()
    if max_cases is not None:
        hit_ids = hit_ids[:int(max_cases)]
    output_paths = []
    for hit_id in hit_ids:
        case = contribution_df[
            contribution_df['Hit_ID'] == hit_id
        ].sort_values('Relative_Position')
        first = case.iloc[0]
        fig, (architecture_ax, logo_ax) = plt.subplots(
            2, 1, figsize=(width, height),
            gridspec_kw={'height_ratios': [0.55, 2.45], 'hspace': 0.08},
        )

        transcript_length = int(first['Transcript_Length'])
        cds_start = int(first['CDS_Start_0based'])
        cds_end = int(first['CDS_End_exclusive'])
        motif_start = int(first['Motif_Start'])
        motif_end = int(first['Motif_End'])
        architecture_ax.plot(
            [0, transcript_length], [0, 0], color='#777777', linewidth=2,
            solid_capstyle='butt',
        )
        architecture_ax.plot(
            [cds_start, cds_end], [0, 0], color='#2E3F83', linewidth=7,
            solid_capstyle='butt',
        )
        architecture_ax.plot(
            [motif_start, motif_end], [0, 0], color='#E67E22', linewidth=9,
            solid_capstyle='butt',
        )
        architecture_ax.text(
            motif_start, 0.22, f"{first['RBP_Name']} motif",
            color='#B85C00', ha='center', va='bottom', fontsize=8,
        )
        architecture_ax.text(
            (cds_start + cds_end) / 2, -0.25, 'CDS',
            color='#2E3F83', ha='center', va='top', fontsize=8,
        )
        architecture_ax.set_xlim(0, transcript_length)
        architecture_ax.set_ylim(-0.45, 0.5)
        architecture_ax.axis('off')

        matrix = _native_contribution_matrix(case)
        logo = logomaker.Logo(
            matrix,
            ax=logo_ax,
            color_scheme={
                'A': '#2CA25F', 'C': '#3B6FB6',
                'G': '#F39C12', 'T': '#D62728',
            },
            shade_below=0.0,
            fade_below=0.0,
            font_name='DejaVu Sans',
        )
        logo.style_spines(visible=False)
        logo.style_spines(spines=['left', 'bottom'], visible=True)
        motif_relative_start = 0
        motif_relative_end = motif_end - motif_start
        logo_ax.axvspan(
            motif_relative_start - 0.5,
            motif_relative_end - 0.5,
            color=motif_color,
            alpha=0.45,
            zorder=-1,
        )
        logo_ax.axhline(0, color='#555555', linewidth=0.7)
        logo_ax.set_xlabel('Position relative to RBP motif start (nt)')
        logo_ax.set_ylabel(r'Base contribution to $\Delta\log_2(TE)$')
        logo_ax.set_title(
            f"{first['Tid']} | {first['RBP_Name']} | {first['Region']} | "
            f"representative hit = {first['Motif_Delta_Log2_TE']:+.3f}\n"
            f"group median = {first['Group_Median_Delta_Log2_TE']:+.3f}, "
            f"n={int(first['Group_N_Transcripts'])} transcripts",
            loc='left', fontsize=9, pad=7,
        )
        logo_ax.grid(axis='y', color='#ECECEC', linewidth=0.5)
        logo_ax.set_axisbelow(True)
        fig.tight_layout()
        filename = (
            f"rbp_base_contribution.{first['RBP_Name']}."
            f"{first['Tid']}.{hit_id}.pdf"
        )
        pdf_path = os.path.join(out_dir, filename)
        fig.savefig(pdf_path, bbox_inches='tight')
        plt.close(fig)
        output_paths.append(pdf_path)
    print(f"Saved {len(output_paths)} RBP nucleotide-contribution cases.")
    return output_paths


def plot_de_novo_translation_motif_logos(
        discovery_df,
        alignments,
        out_path="de_novo_translation_motif_logos.pdf",
        top_n_per_direction=4,
        panel_width=3.0,
        panel_height=1.8):
    """Plot information logos for top signed de novo k-mer enrichments."""
    import logomaker

    required = {
        'Direction', 'Kmer', 'Foreground_Hits', 'Foreground_N',
        'Log2_Enrichment', 'FDR_BH',
    }
    missing = required.difference(discovery_df.columns)
    if missing:
        raise ValueError(
            f"Discovery table is missing columns: {sorted(missing)}"
        )
    selected_parts = []
    for direction in ('Positive', 'Negative'):
        part = discovery_df[discovery_df['Direction'] == direction]
        part = part.sort_values(
            ['FDR_BH', 'Log2_Enrichment'], ascending=[True, False]
        ).head(top_n_per_direction)
        selected_parts.append(part)
    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected[
        selected.apply(
            lambda row: len(alignments.get(
                f"{row['Direction']}|{row['Kmer']}", []
            )) >= 2,
            axis=1,
        )
    ]
    if selected.empty:
        raise ValueError("No de novo motif has enough aligned sequences.")

    n_columns = min(top_n_per_direction, max(
        selected.groupby('Direction', observed=True).size()
    ))
    directions = [
        direction for direction in ('Positive', 'Negative')
        if direction in set(selected['Direction'])
    ]
    fig, axes = plt.subplots(
        len(directions), n_columns,
        figsize=(panel_width * n_columns, panel_height * len(directions)),
        squeeze=False,
    )
    direction_colors = {'Positive': '#C44E52', 'Negative': '#3B6FB6'}
    for row_index, direction in enumerate(directions):
        subset = selected[selected['Direction'] == direction].reset_index(drop=True)
        for column_index in range(n_columns):
            ax = axes[row_index, column_index]
            if column_index >= len(subset):
                ax.axis('off')
                continue
            record = subset.iloc[column_index]
            key = f"{direction}|{record['Kmer']}"
            sequences = list(alignments[key])
            matrix = logomaker.alignment_to_matrix(
                sequences=sequences,
                to_type='information',
            )
            logo = logomaker.Logo(
                matrix, ax=ax, font_name='DejaVu Sans'
            )
            logo.style_spines(visible=False)
            logo.style_spines(spines=['left', 'bottom'], visible=True)
            ax.set_title(
                f"{record['Kmer']} | log2 enrich. "
                f"{record['Log2_Enrichment']:.2f}\n"
                f"FDR={record['FDR_BH']:.2g}, n={len(sequences)}",
                fontsize=8,
                color=direction_colors[direction],
            )
            ax.set_xlabel('Aligned position')
            if column_index == 0:
                ax.set_ylabel(f"{direction}\nInformation (bits)")
            else:
                ax.set_ylabel('')
    fig.tight_layout()
    pdf_path = _as_pdf_path(out_path)
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f"De novo translation-motif logos saved to {pdf_path}")
    return pdf_path
