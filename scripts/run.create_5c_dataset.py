#!/usr/bin/env python3
"""Create five-environment HDF5 datasets by removing HeLa and HEK293T.

Edit the constants below if the datasets are stored elsewhere, then run:

    python run.create_5c_dataset.py

The script preserves the existing HDF5 dataset contract. It copies all sequence
features, copies retained samples without modifying their attributes or target
arrays, keeps only expression vectors used by retained samples, and refreshes
the root sample-count metadata.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterable

import h5py


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR.parent / "dataset"

SOURCE_PREFIX = "human_7c_6k_depth0.1_cov0.1_rpm1"
OUTPUT_PREFIX = "human_5c_6k_depth0.1_cov0.1_rpm1"
SPLITS = ("train", "valid", "test")
REFERENCE_22C_DATASET = (
    DATASET_DIR / "human_tissue_22c_6k_depth0.1_cov0.1_rpm1.train.h5"
)

EXCLUDED_CELL_TYPES = frozenset({"HeLa", "HEK293T"})
EXPECTED_RETAINED_CELL_TYPES = 5
OVERWRITE = False


def _decode_text(value: object) -> str:
    """Return an HDF5 string attribute as a normal Python string."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _copy_attributes(source: h5py.AttributeManager, target: h5py.AttributeManager) -> None:
    """Copy all HDF5 attributes without changing their stored values."""
    for name, value in source.items():
        target[name] = value


def _collect_retained_samples(
    source: h5py.File,
) -> tuple[list[str], Counter[str], Counter[str]]:
    """Collect retained sample identifiers and actual cell-type counts."""
    if "samples" not in source:
        raise KeyError("Input HDF5 file does not contain /samples")

    retained_uuids: list[str] = []
    retained_counts: Counter[str] = Counter()
    excluded_counts: Counter[str] = Counter()

    for uuid, sample in source["samples"].items():
        if "cell_type" not in sample.attrs:
            raise KeyError(f"Sample {uuid!r} has no cell_type attribute")
        cell_type = _decode_text(sample.attrs["cell_type"])
        if cell_type in EXCLUDED_CELL_TYPES:
            excluded_counts[cell_type] += 1
        else:
            retained_uuids.append(str(uuid))
            retained_counts[cell_type] += 1

    source_cell_types = set(retained_counts) | set(excluded_counts)
    missing_exclusions = EXCLUDED_CELL_TYPES - source_cell_types
    if missing_exclusions:
        missing = ", ".join(sorted(missing_exclusions))
        raise ValueError(f"Expected excluded cell types are absent: {missing}")

    if len(retained_counts) != EXPECTED_RETAINED_CELL_TYPES:
        retained = ", ".join(sorted(retained_counts))
        raise ValueError(
            f"Expected {EXPECTED_RETAINED_CELL_TYPES} retained cell types, "
            f"found {len(retained_counts)}: {retained}"
        )

    return retained_uuids, retained_counts, excluded_counts


def _read_sample_cell_types(path: Path) -> set[str]:
    """Read the cell-type set from sample attributes in one HDF5 file."""
    with h5py.File(path, "r") as dataset:
        if "samples" not in dataset:
            raise KeyError(f"Dataset has no /samples group: {path}")
        return {
            _decode_text(sample.attrs.get("cell_type", ""))
            for sample in dataset["samples"].values()
        }


def _copy_filtered_file(source_path: Path, output_path: Path) -> None:
    """Create and validate one filtered HDF5 split."""
    if not source_path.is_file():
        raise FileNotFoundError(f"Input dataset not found: {source_path}")
    if output_path.exists() and not OVERWRITE:
        raise FileExistsError(
            f"Output already exists: {output_path}. Remove it manually or set OVERWRITE=True."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    if temporary_path.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary_path}")

    retained_uuids: list[str]
    retained_counts: Counter[str]
    excluded_counts: Counter[str]

    try:
        with h5py.File(source_path, "r") as source:
            retained_uuids, retained_counts, excluded_counts = _collect_retained_samples(source)

            if "sequences" not in source:
                raise KeyError("Input HDF5 file does not contain /sequences")
            if "cell_exprs" not in source:
                raise KeyError("Input HDF5 file does not contain /cell_exprs")

            missing_expression = set(retained_counts) - set(source["cell_exprs"].keys())
            if missing_expression:
                missing = ", ".join(sorted(missing_expression))
                raise KeyError(f"Missing expression vectors for retained cell types: {missing}")

            with h5py.File(temporary_path, "w") as target:
                _copy_attributes(source.attrs, target.attrs)

                target_samples = target.create_group("samples")
                _copy_attributes(source["samples"].attrs, target_samples.attrs)
                for uuid in retained_uuids:
                    source.copy(source["samples"][uuid], target_samples, name=uuid)

                source.copy(source["sequences"], target, name="sequences")

                target_exprs = target.create_group("cell_exprs")
                _copy_attributes(source["cell_exprs"].attrs, target_exprs.attrs)
                for cell_type in sorted(retained_counts):
                    source.copy(source["cell_exprs"][cell_type], target_exprs, name=cell_type)

                for name in source.keys():
                    if name not in {"samples", "sequences", "cell_exprs"}:
                        source.copy(source[name], target, name=name)

                target.attrs["n_samples"] = len(retained_uuids)
                target.attrs["cell_type_counts"] = json.dumps(
                    dict(sorted(retained_counts.items())), sort_keys=True
                )

        _validate_output(
            temporary_path,
            expected_uuids=retained_uuids,
            expected_counts=retained_counts,
        )

        if output_path.exists() and not OVERWRITE:
            raise FileExistsError(f"Output appeared while processing: {output_path}")
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    removed_summary = ", ".join(
        f"{cell_type}={excluded_counts[cell_type]:,}"
        for cell_type in sorted(EXCLUDED_CELL_TYPES)
    )
    retained_summary = ", ".join(
        f"{cell_type}={retained_counts[cell_type]:,}"
        for cell_type in sorted(retained_counts)
    )
    print(f"Created: {output_path}")
    print(f"  Retained {len(retained_uuids):,} samples: {retained_summary}")
    print(f"  Removed samples: {removed_summary}")


def _validate_output(
    path: Path,
    expected_uuids: Iterable[str],
    expected_counts: Counter[str],
) -> None:
    """Verify metadata, expression vectors, and sample references."""
    expected_uuid_set = set(expected_uuids)

    with h5py.File(path, "r") as dataset:
        required_groups = {"samples", "sequences", "cell_exprs"}
        missing_groups = required_groups - set(dataset.keys())
        if missing_groups:
            missing = ", ".join(sorted(missing_groups))
            raise RuntimeError(f"Generated file is missing required groups: {missing}")

        actual_uuid_set = set(dataset["samples"].keys())
        if actual_uuid_set != expected_uuid_set:
            raise RuntimeError("Generated sample identifiers differ from the retained source samples")

        stored_n_samples = int(dataset.attrs.get("n_samples", -1))
        if stored_n_samples != len(expected_uuid_set):
            raise RuntimeError(
                f"n_samples={stored_n_samples}, expected {len(expected_uuid_set)}"
            )

        stored_counts = json.loads(dataset.attrs.get("cell_type_counts", "{}"))
        if stored_counts != dict(expected_counts):
            raise RuntimeError(
                f"cell_type_counts={stored_counts}, expected {dict(expected_counts)}"
            )

        expression_cells = set(dataset["cell_exprs"].keys())
        if expression_cells != set(expected_counts):
            raise RuntimeError(
                "Expression-vector keys do not match the retained sample cell types"
            )
        if expression_cells & EXCLUDED_CELL_TYPES:
            raise RuntimeError("Excluded expression vectors remain in the generated file")

        actual_counts: Counter[str] = Counter()
        for uuid, sample in dataset["samples"].items():
            cell_type = _decode_text(sample.attrs.get("cell_type", ""))
            tid = _decode_text(sample.attrs.get("tid", ""))
            actual_counts[cell_type] += 1
            if cell_type in EXCLUDED_CELL_TYPES:
                raise RuntimeError(f"Excluded sample remains in output: {uuid}")
            if cell_type not in dataset["cell_exprs"]:
                raise RuntimeError(f"Sample {uuid} has no expression vector for {cell_type}")
            if tid not in dataset["sequences"]:
                raise RuntimeError(f"Sample {uuid} refers to missing sequence {tid}")
            if "count_emb" not in sample:
                raise RuntimeError(f"Sample {uuid} has no count_emb target")

        if actual_counts != expected_counts:
            raise RuntimeError(
                f"Observed sample counts={dict(actual_counts)}, expected {dict(expected_counts)}"
            )


def main() -> None:
    """Generate the train, validation, and test five-cell datasets."""
    print(f"Dataset directory: {DATASET_DIR}")
    print(f"Excluded cell types: {', '.join(sorted(EXCLUDED_CELL_TYPES))}")

    jobs = [
        (
            DATASET_DIR / f"{SOURCE_PREFIX}.{split}.h5",
            DATASET_DIR / f"{OUTPUT_PREFIX}.{split}.h5",
            split,
        )
        for split in SPLITS
    ]

    missing_sources = [str(source) for source, _, _ in jobs if not source.is_file()]
    if missing_sources:
        raise FileNotFoundError(
            "Missing input datasets:\n  " + "\n  ".join(missing_sources)
        )

    if not REFERENCE_22C_DATASET.is_file():
        raise FileNotFoundError(
            f"The 22-cell reference dataset was not found: {REFERENCE_22C_DATASET}"
        )

    existing_outputs = [str(output) for _, output, _ in jobs if output.exists()]
    if existing_outputs and not OVERWRITE:
        raise FileExistsError(
            "Output datasets already exist:\n  "
            + "\n  ".join(existing_outputs)
            + "\nRemove them manually or set OVERWRITE=True."
        )

    reference_cell_types = _read_sample_cell_types(REFERENCE_22C_DATASET)
    retained_sets: dict[str, set[str]] = {}
    for source_path, _, split in jobs:
        source_cell_types = _read_sample_cell_types(source_path)
        retained_cell_types = source_cell_types - EXCLUDED_CELL_TYPES
        retained_sets[split] = retained_cell_types
        missing_from_reference = retained_cell_types - reference_cell_types
        if missing_from_reference:
            missing = ", ".join(sorted(missing_from_reference))
            raise ValueError(
                f"The retained {split} cell types are not a subset of the 22-cell "
                f"reference: {missing}"
            )

    unique_retained_sets = {frozenset(cell_types) for cell_types in retained_sets.values()}
    if len(unique_retained_sets) != 1:
        details = "; ".join(
            f"{split}={sorted(cell_types)}"
            for split, cell_types in retained_sets.items()
        )
        raise ValueError(f"The retained cell types differ across splits: {details}")

    retained_cell_types = next(iter(unique_retained_sets))
    print(f"Retained cell types: {', '.join(sorted(retained_cell_types))}")
    print(f"Verified against: {REFERENCE_22C_DATASET.name}")

    for source_path, output_path, split in jobs:
        print(f"\nProcessing {split}: {source_path.name}")
        _copy_filtered_file(source_path, output_path)

    print("\nAll five-cell datasets were generated and validated successfully.")


if __name__ == "__main__":
    main()
