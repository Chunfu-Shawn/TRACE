"""Tests for incremental RBP database preparation."""

import pickle
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from data.prepare_rbp_database import pre_annotate_and_save_database


def _metadata(rows):
    return pd.DataFrame(
        rows,
        columns=["Matrix_id", "Gene_name", "Gene_id", "Database"],
    )


def test_complete_cache_is_reused_without_api_calls(tmp_path):
    cached_meta = _metadata([
        ("M1", "RBP1", "ENSG000001", "DB1"),
    ])
    cached_meta["RBP_Function"] = "Cached function"
    cached_meta["RBP_GO_BP"] = "Cached process"
    cached_meta.to_csv(
        tmp_path / "Unified_RBP_Metadata_Annotated.tsv",
        sep="\t",
        index=False,
    )
    with open(tmp_path / "Unified_RBP_PWMs.pkl", "wb") as handle:
        pickle.dump({"M1": np.ones((3, 4))}, handle)

    supplied_pwms = {"M1": np.ones((3, 4))}
    with patch("data.prepare_rbp_database.build_opener") as opener_factory:
        result = pre_annotate_and_save_database(
            supplied_pwms,
            cached_meta.drop(columns=["RBP_Function", "RBP_GO_BP"]),
            tmp_path,
            request_delay=0,
        )

    opener_factory.assert_not_called()
    assert len(result) == 1
    assert result.loc[0, "RBP_Function"] == "Cached function"


def test_incremental_merge_fetches_only_new_gene(tmp_path):
    cached_meta = _metadata([
        ("M1", "RBP1", "ENSG000001", "DB1"),
    ])
    cached_meta["RBP_Function"] = "Cached function"
    cached_meta["RBP_GO_BP"] = "Cached process"
    cached_meta.to_csv(
        tmp_path / "Unified_RBP_Metadata_Annotated.tsv",
        sep="\t",
        index=False,
    )
    with open(tmp_path / "Unified_RBP_PWMs.pkl", "wb") as handle:
        pickle.dump({"M1": np.ones((3, 4))}, handle)

    supplied_meta = _metadata([
        ("M1", "RBP1", "ENSG000001", "DB1"),
        ("M2", "RBP2", "ENSG000002", "DB2"),
    ])
    supplied_pwms = {"M1": np.ones((3, 4)), "M2": np.zeros((4, 4))}
    response = Mock()
    response.getcode.return_value = 200
    response.read.return_value = (
        b'{"summary": "New function", "go": {"BP": [{"term": "translation"}]}}'
    )
    opener = Mock()
    opener.open.return_value = response

    with patch("data.prepare_rbp_database.build_opener", return_value=opener):
        result = pre_annotate_and_save_database(
            supplied_pwms,
            supplied_meta,
            tmp_path,
            request_delay=0,
        )

    assert opener.open.call_count == 1
    assert "ENSG000002" in opener.open.call_args.args[0]
    assert set(result["Gene_name"]) == {"RBP1", "RBP2"}
    assert set(supplied_pwms) == {"M1", "M2"}
    assert result.set_index("Gene_name").loc["RBP2", "RBP_Function"] == "New function"


def test_shared_pwm_is_retained_for_multiple_rbps(tmp_path):
    supplied_meta = _metadata([
        ("M_SHARED", "RBP1", "invalid_1", "DB1"),
        ("M_SHARED", "RBP2", "invalid_2", "DB1"),
    ])
    supplied_pwms = {"M_SHARED": np.ones((5, 4))}

    result = pre_annotate_and_save_database(
        supplied_pwms,
        supplied_meta,
        tmp_path,
        request_delay=0,
    )

    assert len(result) == 2
    assert set(result["Gene_name"]) == {"RBP1", "RBP2"}
