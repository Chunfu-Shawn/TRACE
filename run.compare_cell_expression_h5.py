#!/usr/bin/env python3
"""Compare cell-expression vectors stored in four TRACE HDF5 datasets."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Edit these paths before running on the server.
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = Path("/public-supool/home/annie/translation_model/dataset")
OUTPUT_DIR = PROJECT_ROOT.parent / "results/dataset/cell_expression_h5_comparison"

DATASETS = {
    "test_7c": ("human_7c_6k_depth0.1_cov0.1_rpm1.test.h5", 7),
    "tissue_22c": ("human_tissue_22c_6k_depth0.1_cov0.1_rpm1.train.h5", 22),
    "cell_line_18c": ("human_cell_line_18c_6k_depth0.1_cov0.1_rpm1.train.h5", 18),
    "uncommon_26c": (
        "human_cell_line_uncommon_26c_6k_depth0.1_cov0.1_rpm1.train.h5",
        26,
    ),
}

EXPECTED_DIM = 16_840
TARGET_CELLS = ("HeLa", "HEK293T")
TOP_NEIGHBORS = 10


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def decode_json_attr(value) -> dict:
    """Decode a JSON HDF5 attribute."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(str(value))


def load_vectors():
    """Load vectors and perform basic HDF5 integrity checks."""
    vectors = {}
    rows = []
    issues = []

    for dataset_name, (filename, expected_cells) in DATASETS.items():
        path = DATASET_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(path)

        with h5py.File(path, "r") as handle:
            if "cell_exprs" not in handle:
                raise KeyError(f"{path} has no /cell_exprs group")

            sample_counts = decode_json_attr(handle.attrs.get("cell_type_counts", "{}"))
            sample_cells = set(map(str, sample_counts))
            expression_cells = set(map(str, handle["cell_exprs"].keys()))

            if sample_cells != expression_cells:
                issues.append(
                    {
                        "Dataset": dataset_name,
                        "Issue": (
                            f"sample-only={sorted(sample_cells - expression_cells)}; "
                            f"expression-only={sorted(expression_cells - sample_cells)}"
                        ),
                    }
                )
            if len(expression_cells) != expected_cells:
                issues.append(
                    {
                        "Dataset": dataset_name,
                        "Issue": (
                            f"found {len(expression_cells)} cell types; "
                            f"expected {expected_cells}"
                        ),
                    }
                )
            if int(handle.attrs.get("n_samples", -1)) != sum(sample_counts.values()):
                issues.append(
                    {
                        "Dataset": dataset_name,
                        "Issue": "n_samples differs from the sum of cell_type_counts",
                    }
                )

            for cell_type in sorted(expression_cells):
                stored = handle["cell_exprs"][cell_type]
                vector = np.asarray(stored[:], dtype=np.float64).reshape(-1)
                label = f"{cell_type} [{dataset_name}]"
                vectors[label] = vector
                rows.append(
                    {
                        "Dataset": dataset_name,
                        "Cell_Type": cell_type,
                        "Label": label,
                        "Sample_Count": int(sample_counts.get(cell_type, 0)),
                        "Dimension": int(vector.size),
                        "Stored_Dtype": str(stored.dtype),
                        "Finite": bool(np.isfinite(vector).all()),
                        "Mean": float(np.mean(vector)),
                        "Std": float(np.std(vector)),
                        "Zero_Fraction": float(np.mean(vector == 0.0)),
                        "Min": float(np.min(vector)),
                        "Max": float(np.max(vector)),
                    }
                )

                if vector.size != EXPECTED_DIM:
                    issues.append(
                        {
                            "Dataset": dataset_name,
                            "Issue": f"{cell_type}: dimension={vector.size}",
                        }
                    )
                if not np.isfinite(vector).all():
                    issues.append(
                        {"Dataset": dataset_name, "Issue": f"{cell_type}: non-finite values"}
                    )
                if float(np.std(vector)) < 1e-8:
                    issues.append(
                        {"Dataset": dataset_name, "Issue": f"{cell_type}: constant vector"}
                    )

    return vectors, pd.DataFrame(rows), issues


def check_duplicate_assignments(vectors, metadata, issues):
    """Detect inconsistent repeated names and identical different-cell vectors."""
    labels = metadata["Label"].tolist()
    cells = metadata.set_index("Label")["Cell_Type"].to_dict()
    for first_index, first_label in enumerate(labels):
        for second_label in labels[first_index + 1 :]:
            first = vectors[first_label]
            second = vectors[second_label]
            if first.shape != second.shape:
                continue
            identical = np.array_equal(first, second)
            if cells[first_label] == cells[second_label] and not identical:
                issues.append(
                    {
                        "Dataset": "cross_file",
                        "Issue": (
                            f"same cell name has different vectors: {first_label} / "
                            f"{second_label}"
                        ),
                    }
                )
            if cells[first_label] != cells[second_label] and identical:
                issues.append(
                    {
                        "Dataset": "cross_file",
                        "Issue": (
                            f"different cells have identical vectors: {first_label} / "
                            f"{second_label}; check mean-vector fallback"
                        ),
                    }
                )


def nearest_neighbors(correlation, metadata):
    """Return the top Pearson neighbors for HeLa and HEK293T."""
    rows = []
    target_rows = metadata[metadata["Cell_Type"].isin(TARGET_CELLS)]
    for _, target in target_rows.iterrows():
        target_label = target["Label"]
        ranked = correlation.loc[target_label].drop(target_label).sort_values(ascending=False)
        for rank, (neighbor, value) in enumerate(ranked.head(TOP_NEIGHBORS).items(), 1):
            rows.append(
                {
                    "Target": target_label,
                    "Rank": rank,
                    "Neighbor": neighbor,
                    "Pearson": float(value),
                }
            )
    return pd.DataFrame(rows)


def save_heatmap(correlation):
    """Save a clustered Pearson-correlation heatmap."""
    size = max(12.0, min(22.0, len(correlation) * 0.24))
    grid = sns.clustermap(
        correlation,
        cmap="vlag",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        figsize=(size, size),
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"label": "Pearson correlation"},
    )
    grid.ax_heatmap.tick_params(axis="both", labelsize=5)
    grid.fig.suptitle("Cell-expression vectors stored in TRACE HDF5 datasets", y=1.01)
    output_stem = OUTPUT_DIR / "cell_expression_pearson_heatmap"
    grid.fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    grid.fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    grid.fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(grid.fig)


def main():
    """Run the comparison and write compact tabular and visual outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vectors, metadata, issues = load_vectors()
    check_duplicate_assignments(vectors, metadata, issues)

    issue_table = pd.DataFrame(issues, columns=["Dataset", "Issue"])
    metadata.to_csv(OUTPUT_DIR / "expression_vector_qc.csv", index=False)
    issue_table.to_csv(OUTPUT_DIR / "integrity_issues.csv", index=False)

    labels = metadata["Label"].tolist()
    dimensions = {vectors[label].size for label in labels}
    vectors_are_valid = all(
        np.isfinite(vectors[label]).all() and float(np.std(vectors[label])) >= 1e-8
        for label in labels
    )
    if dimensions != {EXPECTED_DIM} or not vectors_are_valid:
        raise ValueError(f"Invalid vectors found; inspect {OUTPUT_DIR / 'integrity_issues.csv'}")

    matrix = np.stack([vectors[label] for label in labels])
    correlation = pd.DataFrame(np.corrcoef(matrix), index=labels, columns=labels)
    neighbors = nearest_neighbors(correlation, metadata)

    correlation.to_csv(OUTPUT_DIR / "pairwise_pearson.csv")
    neighbors.to_csv(OUTPUT_DIR / "target_nearest_neighbors.csv", index=False)
    save_heatmap(correlation)

    print(f"Loaded {len(metadata)} expression vectors from {len(DATASETS)} HDF5 files.")
    print(metadata.groupby("Dataset")["Cell_Type"].count().to_string())
    print("\nHeLa/HEK293T nearest neighbors:")
    print(neighbors.to_string(index=False))
    if issues:
        print("\nIntegrity warnings:")
        print(issue_table.to_string(index=False))
    else:
        print("\nPASS: no structural, dimensional, finite-value, or duplicate-vector issues.")
    print(
        "\nImportant: these HDF5 files store vectors but not gene IDs or the gene-order "
        "checksum. This audit cannot prove historical gene order from HDF5 alone."
    )
    print(f"\nResults: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
