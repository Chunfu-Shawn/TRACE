#!/usr/bin/env python3
"""Inspect target keys and possible aliases in a TRACE expression dictionary."""

from __future__ import annotations

import difflib
import hashlib
import re
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
EXPRESSION_DICT_PATH = PROJECT_ROOT / "src/config/human_expression_dict.pt"
TARGET_KEYS = (
    "skeletal_muscle_reduced_activity",
    "B721.221",
    "U-251",
    "U-343",
)


def normalize_key(value: str) -> str:
    """Normalize punctuation and case for diagnostic matching only."""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def tensor_stats(value) -> str:
    """Return compact numerical statistics for one expression vector."""
    tensor = torch.as_tensor(value).detach().float().reshape(-1)
    return (
        f"shape={tuple(tensor.shape)}, mean={tensor.mean().item():.6f}, "
        f"std={tensor.std(unbiased=False).item():.6f}, "
        f"min={tensor.min().item():.6f}, max={tensor.max().item():.6f}"
    )


def file_sha256(path: Path) -> str:
    """Calculate a reproducible checksum for the dictionary file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Print exact lookup results and likely naming mismatches."""
    if not EXPRESSION_DICT_PATH.is_file():
        raise FileNotFoundError(EXPRESSION_DICT_PATH)

    expression_dict = torch.load(EXPRESSION_DICT_PATH, map_location="cpu")
    if not isinstance(expression_dict, dict):
        raise TypeError(f"Expected a dictionary, found {type(expression_dict).__name__}")

    keys = [str(key) for key in expression_dict]
    normalized_to_keys = {}
    for key in keys:
        normalized_to_keys.setdefault(normalize_key(key), []).append(key)

    print(f"Expression dictionary: {EXPRESSION_DICT_PATH}")
    print(f"SHA-256: {file_sha256(EXPRESSION_DICT_PATH)}")
    print(f"Number of keys: {len(keys)}")

    resolved = {}
    missing_exact = []
    for target in TARGET_KEYS:
        exact = target in expression_dict
        normalized_matches = normalized_to_keys.get(normalize_key(target), [])
        prefix_candidate = target.rsplit(".", 1)[0] if "." in target else None
        prefix_match = prefix_candidate if prefix_candidate in expression_dict else None
        close_matches = difflib.get_close_matches(target, keys, n=5, cutoff=0.45)

        print(f"\nTarget: {target!r}")
        print(f"  Exact key: {exact}")
        print(f"  DatasetGenerator fallback: {not exact}")
        print(f"  Normalized matches: {normalized_matches or 'none'}")
        print(f"  Prefix match: {prefix_match or 'none'}")
        print(f"  Closest keys: {close_matches or 'none'}")

        if exact:
            resolved[target] = target
            print(f"  Vector: {tensor_stats(expression_dict[target])}")
        else:
            missing_exact.append(target)
            if len(normalized_matches) == 1:
                candidate = normalized_matches[0]
                resolved[target] = candidate
                print(f"  Candidate vector: {tensor_stats(expression_dict[candidate])}")
            elif prefix_match is not None:
                resolved[target] = prefix_match
                print(f"  Candidate vector: {tensor_stats(expression_dict[prefix_match])}")

    print("\nPairwise equality among resolved target vectors:")
    resolved_items = list(resolved.items())
    for first_index, (first_target, first_key) in enumerate(resolved_items):
        first = torch.as_tensor(expression_dict[first_key]).detach().cpu().reshape(-1)
        for second_target, second_key in resolved_items[first_index + 1 :]:
            second = torch.as_tensor(expression_dict[second_key]).detach().cpu().reshape(-1)
            if first.shape != second.shape:
                comparison = "different shapes"
            else:
                comparison = (
                    f"equal={torch.equal(first, second)}, "
                    f"max_abs_diff={(first.float() - second.float()).abs().max().item():.6f}"
                )
            print(f"  {first_target!r} vs {second_target!r}: {comparison}")

    print("\nSummary:")
    if missing_exact:
        print(f"  Missing exact keys: {missing_exact}")
        print("  These names would trigger mean_expr_vector fallback in DatasetGenerator.")
    else:
        print("  All target keys exist exactly; none would trigger fallback.")

    if "B721.221" in missing_exact and "B721" in expression_dict:
        print("  Likely mismatch: dataset cell_type='B721.221', dictionary key='B721'.")
    if "U-251" in expression_dict and "U-343" in expression_dict:
        same = torch.equal(
            torch.as_tensor(expression_dict["U-251"]),
            torch.as_tensor(expression_dict["U-343"]),
        )
        print(f"  Current U-251 and U-343 vectors are identical: {same}")
        if not same:
            print(
                "  If an existing HDF5 stores them identically, it was likely built "
                "with an older or different expression dictionary."
            )


if __name__ == "__main__":
    main()
