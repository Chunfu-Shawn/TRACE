#!/usr/bin/env python3
"""Shared normalization utilities for complete-GTF featureCounts matrices."""

import os

import pandas as pd


FEATURECOUNTS_METADATA_COLUMNS = ['Geneid', 'Chr', 'Start', 'End', 'Strand', 'Length']


def clean_featurecounts_sample_name(column):
    """Convert a featureCounts BAM column into its sample/run identifier."""
    basename = os.path.basename(column) if '.bam' in column else column
    for suffix in ['.uniq.sorted.bam', '_uniq.sorted.bam', '.bam']:
        if basename.endswith(suffix):
            return basename[:-len(suffix)]
    return basename


def calculate_true_tpm(counts, lengths):
    """Calculate TPM using each sample's sum of transcript RPK as denominator."""
    numeric_counts = counts.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    numeric_lengths = pd.to_numeric(lengths, errors='coerce')
    if numeric_lengths.isna().any() or (numeric_lengths <= 0).any():
        bad_ids = numeric_lengths.index[numeric_lengths.isna() | (numeric_lengths <= 0)].tolist()
        raise ValueError(f"Transcripts have invalid effective lengths: {bad_ids[:5]}")

    rpk = numeric_counts.div(numeric_lengths / 1000.0, axis=0)
    scale_factors = rpk.sum(axis=0) / 1e6
    if (scale_factors <= 0).any():
        bad_samples = scale_factors.index[scale_factors <= 0].tolist()
        raise ValueError(f"Samples have zero total transcript RPK: {bad_samples}")
    return rpk.div(scale_factors, axis=1)


def calculate_gene_read_library_sizes(gene_counts_file):
    """Sum complete-GTF gene counts per sample for junction CPM denominators."""
    gene_df = pd.read_csv(gene_counts_file, sep='\t', comment='#')
    gene_df.rename(columns=clean_featurecounts_sample_name, inplace=True)
    count_cols = [
        column for column in gene_df.columns
        if column not in FEATURECOUNTS_METADATA_COLUMNS
    ]
    if not count_cols:
        raise ValueError("Gene counts file contains no sample columns.")

    numeric_counts = gene_df[count_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    library_sizes = numeric_counts.sum(axis=0)
    if (library_sizes <= 0).any():
        bad_samples = library_sizes.index[library_sizes <= 0].tolist()
        raise ValueError(f"Samples have zero gene-read library size: {bad_samples}")
    return library_sizes
