"""Prepare and incrementally annotate local RBP motif databases."""

from __future__ import annotations

import json
import os
import pickle
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import build_opener

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - used only in minimal environments
    def tqdm(iterable, **kwargs):
        """Fall back to a plain iterator when tqdm is unavailable."""
        return iterable


METADATA_FILENAME = "Unified_RBP_Metadata_Annotated.tsv"
PWM_FILENAME = "Unified_RBP_PWMs.pkl"


def _is_present(value: Any) -> bool:
    """Return whether a metadata value contains usable information."""
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return False
    return bool(str(value).strip())


def _first_present(values: pd.Series) -> Any:
    """Keep the first non-empty value while merging cached and new rows."""
    for value in values:
        if _is_present(value):
            return value
    return pd.NA


def _metadata_identity_columns(metadata: pd.DataFrame) -> list[str]:
    """Return columns identifying an RBP-to-motif association."""
    columns = ["Matrix_id"]
    for column in ("Gene_id", "Gene_name", "Database"):
        if column in metadata.columns:
            columns.append(column)
    return columns


def _metadata_identity_set(metadata: pd.DataFrame) -> set[tuple]:
    """Build normalized association keys for cache-completeness checks."""
    if metadata.empty or "Matrix_id" not in metadata.columns:
        return set()
    key_columns = _metadata_identity_columns(metadata)
    normalized = metadata[key_columns].fillna("").astype(str)
    return set(normalized.itertuples(index=False, name=None))


def _merge_metadata(cached_meta: pd.DataFrame, new_meta: pd.DataFrame) -> pd.DataFrame:
    """Merge metadata without losing RBPs that share the same motif matrix."""
    if "Matrix_id" not in new_meta.columns and "Matrix_id" not in cached_meta.columns:
        raise ValueError("combined_meta must contain a 'Matrix_id' column.")

    frames = []
    for frame in (cached_meta, new_meta):
        if frame is None or frame.empty:
            continue
        frame = frame.copy()
        if "Matrix_id" not in frame.columns:
            raise ValueError("All metadata tables must contain a 'Matrix_id' column.")
        frame["Matrix_id"] = frame["Matrix_id"].astype(str).str.strip()
        frame = frame[frame["Matrix_id"].ne("") & frame["Matrix_id"].ne("nan")]
        frames.append(frame)

    if not frames:
        return new_meta.copy()

    merged = pd.concat(frames, ignore_index=True, sort=False)
    column_order = list(dict.fromkeys(
        list(cached_meta.columns) + list(new_meta.columns)
    ))
    identity_columns = _metadata_identity_columns(merged)
    value_columns = [
        column for column in merged.columns if column not in identity_columns
    ]
    if value_columns:
        merged = (
            merged.groupby(identity_columns, sort=False, as_index=False, dropna=False)
            .agg({column: _first_present for column in value_columns})
        )
    else:
        merged = merged.drop_duplicates(identity_columns, keep="first")
    return merged.reindex(columns=column_order)


def _load_cached_database(meta_path: str, pwm_path: str):
    """Load whichever cache files are available."""
    cached_meta = pd.DataFrame()
    cached_pwms = {}

    if os.path.isfile(meta_path):
        cached_meta = pd.read_csv(meta_path, sep="\t", dtype={"Matrix_id": str})
        print(f"Loaded cached metadata: {meta_path} ({len(cached_meta)} motifs)")

    if os.path.isfile(pwm_path):
        with open(pwm_path, "rb") as handle:
            cached_pwms = pickle.load(handle)
        if not isinstance(cached_pwms, dict):
            raise TypeError(f"Cached PWM file must contain a dictionary: {pwm_path}")
        cached_pwms = {str(key).strip(): value for key, value in cached_pwms.items()}
        print(f"Loaded cached PWMs: {pwm_path} ({len(cached_pwms)} matrices)")

    return cached_meta, cached_pwms


def _annotation_is_reusable(value: Any) -> bool:
    """Reject empty or transient API-error annotations so they can be retried."""
    if not _is_present(value):
        return False
    text = str(value).strip()
    return text != "API Fetch Error" and not text.startswith("HTTP ")


def _build_annotation_cache(metadata: pd.DataFrame, column: str) -> dict:
    """Recover reusable gene-level annotations from cached motif metadata."""
    if "Gene_id" not in metadata.columns or column not in metadata.columns:
        return {}
    cache = {}
    for gene_id, value in zip(metadata["Gene_id"], metadata[column]):
        if _is_present(gene_id) and _annotation_is_reusable(value):
            cache.setdefault(str(gene_id).strip(), value)
    return cache


def _fetch_gene_annotation(gene_id: str, opener, request_delay: float):
    """Fetch the functional summary and GO biological-process terms for one gene."""
    if not gene_id.startswith("ENSG"):
        return "Unannotated (Invalid ID)", "None"

    url = f"https://mygene.info/v3/gene/{gene_id}?fields=summary,name,go.BP"
    try:
        if request_delay > 0:
            time.sleep(request_delay)
        response = opener.open(url, timeout=5)
        status_code = response.getcode()
        if status_code != 200:
            return f"HTTP {status_code}", "None"
        data = json.loads(response.read().decode("utf-8"))
        function = data.get("summary", data.get("name", "Summary unavailable in NCBI."))
        bp_entries = data.get("go", {}).get("BP", [])
        if isinstance(bp_entries, dict):
            bp_entries = [bp_entries]
        bp_terms = sorted({
            entry.get("term")
            for entry in bp_entries
            if isinstance(entry, dict) and entry.get("term")
        })
        go_bp = "; ".join(bp_terms) if bp_terms else "No BP terms annotated"
        return function, go_bp
    except HTTPError as error:
        if error.code == 404:
            return "Gene not found in MyGene.", "None"
        return f"HTTP {error.code}", "None"
    except Exception:
        return "API Fetch Error", "None"


def pre_annotate_and_save_database(
    combined_pwms,
    combined_meta,
    out_dir,
    request_delay=0.1,
):
    """Incrementally annotate and persist a unified RBP motif database.

    Existing ``Unified_RBP_PWMs.pkl`` and
    ``Unified_RBP_Metadata_Annotated.tsv`` files are treated as a cache. New
    motif matrices and metadata rows are merged by ``Matrix_id``. Gene
    annotations already present in the cache are reused, and only new or
    incompletely annotated Ensembl gene IDs are queried from MyGene.info.
    """
    if not isinstance(combined_pwms, dict):
        raise TypeError("combined_pwms must be a dictionary keyed by Matrix_id.")
    if not isinstance(combined_meta, pd.DataFrame):
        raise TypeError("combined_meta must be a pandas DataFrame.")
    if "Matrix_id" not in combined_meta.columns:
        raise ValueError("combined_meta must contain a 'Matrix_id' column.")
    if "Gene_id" not in combined_meta.columns:
        raise ValueError("combined_meta must contain a 'Gene_id' column.")

    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, METADATA_FILENAME)
    pwm_path = os.path.join(out_dir, PWM_FILENAME)
    cached_meta, cached_pwms = _load_cached_database(meta_path, pwm_path)

    new_pwms = {str(key).strip(): value for key, value in combined_pwms.items()}
    unified_pwms = {**cached_pwms, **new_pwms}
    unified_meta = _merge_metadata(cached_meta, combined_meta)

    identity_columns = _metadata_identity_columns(unified_meta)
    cached_for_coverage = cached_meta.reindex(columns=identity_columns)
    new_for_coverage = combined_meta.reindex(columns=identity_columns)
    missing_meta_associations = (
        _metadata_identity_set(new_for_coverage)
        - _metadata_identity_set(cached_for_coverage)
    )
    missing_pwm_ids = sorted(set(new_pwms) - set(cached_pwms))
    print(
        "Cache coverage: "
        f"{len(missing_meta_associations)} new RBP-motif associations, "
        f"{len(missing_pwm_ids)} new PWM matrices."
    )

    summary_cache = _build_annotation_cache(cached_meta, "RBP_Function")
    go_bp_cache = _build_annotation_cache(cached_meta, "RBP_GO_BP")
    gene_ids = [
        str(value).strip()
        for value in unified_meta["Gene_id"].dropna().unique()
        if str(value).strip()
    ]
    genes_to_fetch = [
        gene_id
        for gene_id in gene_ids
        if gene_id not in summary_cache or gene_id not in go_bp_cache
    ]

    if genes_to_fetch:
        print(f"Fetching annotations for {len(genes_to_fetch)} new/incomplete genes.")
        opener = build_opener()
        for gene_id in tqdm(genes_to_fetch, desc="Fetching MyGene annotations"):
            function, go_bp = _fetch_gene_annotation(gene_id, opener, request_delay)
            summary_cache[gene_id] = function
            go_bp_cache[gene_id] = go_bp
    else:
        print("All supplied RBPs are already annotated; no API requests are needed.")

    gene_key = unified_meta["Gene_id"].map(
        lambda value: str(value).strip() if _is_present(value) else None
    )
    unified_meta["RBP_Function"] = gene_key.map(summary_cache)
    unified_meta["RBP_GO_BP"] = gene_key.map(go_bp_cache)

    # Use atomic replacement so an interrupted write does not corrupt the cache.
    meta_tmp_path = f"{meta_path}.tmp"
    pwm_tmp_path = f"{pwm_path}.tmp"
    unified_meta.to_csv(meta_tmp_path, sep="\t", index=False)
    with open(pwm_tmp_path, "wb") as handle:
        pickle.dump(unified_pwms, handle)
    os.replace(meta_tmp_path, meta_path)
    os.replace(pwm_tmp_path, pwm_path)

    # Keep caller-owned dictionaries synchronized with matrices recovered from cache.
    combined_pwms.clear()
    combined_pwms.update(unified_pwms)

    print(f"Annotated metadata saved: {meta_path} ({len(unified_meta)} motifs)")
    print(f"Unified PWM dictionary saved: {pwm_path} ({len(unified_pwms)} matrices)")
    return unified_meta
