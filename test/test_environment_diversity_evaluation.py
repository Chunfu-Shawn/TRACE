"""Tests for partial-grid execution and persistent evaluation caches."""

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def load_script_module():
    """Load the run script while stubbing optional inference-only imports."""
    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda iterable, *args, **kwargs: iterable

    periodicity_module = types.ModuleType("eval.periodicity_corr")
    periodicity_module.calculate_periodicity = lambda *args, **kwargs: np.nan

    prediction_module = types.ModuleType("eval.save_prediction_results")
    prediction_module.save_count_predictions = lambda *args, **kwargs: "unused.pkl"

    base_model_module = types.ModuleType("model.base_model")
    base_model_module.BaseModel = object

    prediction_head_module = types.ModuleType("model.prediction_heads")
    prediction_head_module.PsiteDensityHead = object

    module_name = "environment_diversity_under_test"
    spec = importlib.util.spec_from_file_location(
        module_name,
        PROJECT_ROOT / "run.evaluate_environment_diversity.py",
    )
    module = importlib.util.module_from_spec(spec)
    temporary_modules = {
        "tqdm": tqdm_module,
        "eval.periodicity_corr": periodicity_module,
        "eval.save_prediction_results": prediction_module,
        "model.base_model": base_model_module,
        "model.prediction_heads": prediction_head_module,
        module_name: module,
    }
    missing = object()
    previous_modules = {
        name: sys.modules.get(name, missing) for name in temporary_modules
    }
    sys.modules.update(temporary_modules)
    try:
        spec.loader.exec_module(module)
    finally:
        for name, previous in previous_modules.items():
            if previous is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


environment_diversity = load_script_module()


class EnvironmentDiversityEvaluationTests(unittest.TestCase):
    def test_cache_only_checkpoint_must_match_the_configured_rule(self):
        spec = environment_diversity.ModelSpec(
            environment_count=40,
            strategy="exp_aug",
            checkpoint=None,
            config_path=Path("model.yaml"),
        )
        matching = {
            "checkpoint": {
                "path": (
                    "/checkpoints/base_model_hs_22c_18c_"
                    "a2_b02_exp_aug.best_profile.pt"
                )
            }
        }
        wrong_selection = {
            "checkpoint": {
                "path": (
                    "/checkpoints/base_model_hs_22c_18c_"
                    "a2_b02_exp_aug.best_total.pt"
                )
            }
        }

        self.assertTrue(
            environment_diversity.cached_checkpoint_matches_spec(spec, matching)
        )
        self.assertFalse(
            environment_diversity.cached_checkpoint_matches_spec(
                spec, wrong_selection
            )
        )

    def test_missing_checkpoint_returns_an_empty_grid_position(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                environment_diversity,
                "CHECKPOINT_DIR",
                Path(directory),
            ), patch.object(
                environment_diversity,
                "CHECKPOINT_MATCH_RULES",
                {(5, "zero"): {"required": ("missing",), "forbidden": ()}},
            ), patch.object(environment_diversity, "EXACT_CHECKPOINTS", {}):
                checkpoint, status, message = environment_diversity.resolve_checkpoint(
                    5, "zero"
                )

        self.assertIsNone(checkpoint)
        self.assertEqual(status, "missing_checkpoint")
        self.assertIn("No checkpoint matched", message)

    def test_prediction_cache_requires_matching_fingerprints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "results"
            prediction_dir = output_dir / "predictions"
            prediction_dir.mkdir(parents=True)
            dataset_path = root / "test.h5"
            config_path = root / "model.yaml"
            checkpoint_path = root / "model.pt"
            dataset_path.write_bytes(b"dataset")
            config_path.write_text("model: test\n", encoding="utf-8")
            checkpoint_path.write_bytes(b"checkpoint-v1")
            spec = environment_diversity.ModelSpec(
                environment_count=5,
                strategy="zero",
                checkpoint=checkpoint_path,
                config_path=config_path,
            )
            prediction_path = prediction_dir / "predictions_count.base.5c_zero.pkl"
            prediction_path.write_bytes(b"prediction")

            with patch.object(environment_diversity, "OUTPUT_DIR", output_dir), patch.object(
                environment_diversity,
                "TEST_DATASET_PATH",
                dataset_path,
            ):
                manifest = environment_diversity.prediction_manifest(spec)
                Path(str(prediction_path) + ".manifest.json").write_text(
                    json.dumps({**manifest, "checkpoint_epoch": 12}),
                    encoding="utf-8",
                )
                cached = environment_diversity.find_cached_prediction(spec)
                checkpoint_path.write_bytes(b"checkpoint-v2-changed")
                stale = environment_diversity.find_cached_prediction(spec)

        self.assertEqual(cached, prediction_path)
        self.assertIsNone(stale)

    def test_missing_models_remain_in_summary_and_plots(self):
        cells = ["cell_a", "cell_b"]
        specs = [
            environment_diversity.ModelSpec(
                environment_count=count,
                strategy=strategy,
                checkpoint=None,
                config_path=Path("missing.yaml"),
                resolution_status="missing_checkpoint",
            )
            for count in (5, 22, 40)
            for strategy in ("zero", "real", "exp_aug")
        ]
        cell_metrics = pd.concat(
            [
                environment_diversity.empty_cell_metrics(spec, cells)
                for spec in specs
            ],
            ignore_index=True,
        )
        cell_metrics["Nearest_Training_Cell"] = ""
        cell_metrics["Nearest_Cosine_Distance"] = np.nan
        cell_metrics["Expression_Coverage"] = np.nan
        model_metrics = environment_diversity.summarize_models(cell_metrics)

        with patch.object(
            environment_diversity,
            "save_publication_figure",
        ) as save_figure:
            environment_diversity.plot_environment_diversity_curves(
                model_metrics,
                Path("environment"),
            )
            regression = environment_diversity.plot_zero_shot_distance_curves(
                cell_metrics,
                Path("distance"),
            )

        self.assertEqual(len(model_metrics), 9)
        self.assertTrue((model_metrics["RNA_N"] == 0).all())
        self.assertTrue(model_metrics["Mean_CDS_Profile_Spearman"].isna().all())
        self.assertEqual(len(regression), 6)
        self.assertEqual(save_figure.call_count, 2)

    def test_transcript_metrics_are_reused_without_re_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "results"
            dataset_path = root / "test.h5"
            prediction_path = root / "prediction.pkl"
            dataset_path.write_bytes(b"dataset")
            prediction_path.write_bytes(b"prediction")
            spec = environment_diversity.ModelSpec(
                environment_count=22,
                strategy="real",
                checkpoint=None,
                config_path=root / "missing.yaml",
            )
            row = {
                column: np.nan
                for column in environment_diversity.TRANSCRIPT_METRIC_COLUMNS
            }
            row.update(
                {
                    "Model_ID": spec.model_id,
                    "Environment_Count": 22,
                    "Strategy": "real",
                    "Strategy_Label": spec.strategy_label,
                    "Cell_Type": "cell_a",
                }
            )

            with patch.object(environment_diversity, "OUTPUT_DIR", output_dir), patch.object(
                environment_diversity,
                "TEST_DATASET_PATH",
                dataset_path,
            ):
                csv_path, manifest_path = environment_diversity.metric_cache_paths(
                    prediction_path
                )
                pd.DataFrame([row]).to_csv(csv_path, index=False)
                manifest_path.write_text(
                    json.dumps(
                        environment_diversity.metric_manifest(prediction_path, spec)
                    ),
                    encoding="utf-8",
                )
                with patch.object(
                    environment_diversity,
                    "evaluate_prediction_file",
                    side_effect=AssertionError("metrics should be reused"),
                ):
                    metrics, observed_path, reused = (
                        environment_diversity.load_or_evaluate_metrics(
                            object(), prediction_path, spec
                        )
                    )

        self.assertTrue(reused)
        self.assertEqual(observed_path, csv_path)
        self.assertEqual(len(metrics), 1)


if __name__ == "__main__":
    unittest.main()
