"""Tests for matched CDS-start Kozak mutagenesis."""

import sys
import types
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import tqdm  # noqa: F401
except ModuleNotFoundError:
    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda iterable=None, **kwargs: iterable
    sys.modules["tqdm"] = tqdm_module

try:
    import yaml  # noqa: F401
except ModuleNotFoundError:
    yaml_module = types.ModuleType("yaml")
    yaml_module.safe_load = lambda value: value
    yaml_module.safe_dump = lambda value, stream: None
    sys.modules["yaml"] = yaml_module

from eval.start_codon_kozak_mutagenesis import (
    KOZAK_CONTEXT_ORDER,
    KozakMutagenesisEvaluator,
    _p_site_intensity,
    collect_kozak_mutagenesis_samples,
    evaluate_start_codon_kozak_mutagenesis,
    mutate_cds_start_context,
    plot_kozak_mutagenesis_results,
)
from model.prediction_heads import PsiteDensityHead
from model.translation_base_model import TranslationBaseModel


class SequenceSensitiveModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def predict(self, **kwargs):
        sequence = kwargs["seq_batch"]
        signal = torch.ones(
            sequence.shape[0], sequence.shape[1], 1, device=sequence.device
        )
        return {"count": torch.log1p(signal)}


def _make_dataset():
    sequence = np.zeros((60, 4), dtype=np.float32)
    sequence[:, 3] = 1.0
    return [(
        "ENST000001.4-sample",
        "human",
        "brain",
        torch.zeros(3),
        {"cds_start_pos": 13, "cds_end_pos": 48},
        torch.from_numpy(sequence),
        torch.ones(60, 1),
    )]


class KozakMutagenesisTests(unittest.TestCase):
    def test_top_level_reuses_raw_csv_without_model_or_dataset(self):
        cached = pd.DataFrame({"P_Site_Intensity": [0.25]})
        expected_paths = {
            "boxplot": "boxplot.pdf",
            "per_codon_scatter": "per_codon.pdf",
            "global_scatter": "global.pdf",
        }
        with TemporaryDirectory() as temporary_directory:
            raw_path = (
                Path(temporary_directory)
                / "kozak_mutagenesis_raw.cached.csv"
            )
            cached.to_csv(raw_path, index=False)
            with patch(
                "eval.start_codon_kozak_mutagenesis."
                "plot_kozak_mutagenesis_results",
                return_value=expected_paths,
            ) as plot_mock:
                results, paths = evaluate_start_codon_kozak_mutagenesis(
                    model=object(),
                    dataset=object(),
                    out_dir=temporary_directory,
                    suffix="cached",
                )

        pd.testing.assert_frame_equal(results, cached)
        self.assertEqual(paths, expected_paths)
        plot_mock.assert_called_once()

    def test_p_site_intensity_matches_motif_definition(self):
        profile = np.arange(1, 13, dtype=np.float32)
        intensity, p_site, local_sum, global_mean = _p_site_intensity(
            profile,
            start=6,
        )
        expected_global_mean = np.mean(profile) + 1e-6
        expected_local_sum = np.sum(profile[3:9])
        expected_intensity = profile[6] / (
            expected_local_sum + expected_global_mean
        )

        self.assertAlmostEqual(p_site, float(profile[6]))
        self.assertAlmostEqual(local_sum, float(expected_local_sum))
        self.assertAlmostEqual(global_mean, float(expected_global_mean))
        self.assertAlmostEqual(intensity, float(expected_intensity))

    def test_translation_model_uses_zero_count_when_omitted(self):
        model = TranslationBaseModel(
            d_seq=4,
            d_count=1,
            d_model=8,
            d_expr=3,
            d_cell_env=4,
            all_species=["human"],
            d_species=2,
            n_heads=2,
            number_of_layers=1,
            d_ff=16,
            adaptive_dim=4,
            p_drop=0.0,
        ).eval()
        sequence = torch.nn.functional.one_hot(
            torch.arange(18).reshape(2, 9) % 4,
            num_classes=4,
        ).float()
        expression = torch.zeros(2, 3)
        mask = torch.ones(2, 9, dtype=torch.bool)

        implicit = model(
            seq_batch=sequence,
            expr_vector=expression,
            species=["human", "human"],
            src_mask=mask,
        )
        explicit = model(
            seq_batch=sequence,
            count_batch=torch.zeros(2, 9, 1),
            expr_vector=expression,
            species=["human", "human"],
            src_mask=mask,
        )

        torch.testing.assert_close(implicit, explicit)

    def test_evaluator_accepts_translation_model_without_count_input(self):
        model = TranslationBaseModel(
            d_seq=4,
            d_count=1,
            d_model=8,
            d_expr=3,
            d_cell_env=4,
            all_species=["human"],
            d_species=2,
            n_heads=2,
            number_of_layers=1,
            d_ff=16,
            adaptive_dim=4,
            p_drop=0.0,
        )
        model.add_head(
            "count",
            PsiteDensityHead.create_from_model(
                model,
                d_count=1,
                d_pred_h=8,
                p_drop=0.0,
            ),
            overwrite=True,
            move_to_model_device=False,
        )
        samples = collect_kozak_mutagenesis_samples(_make_dataset())
        evaluator = KozakMutagenesisEvaluator(
            model,
            prediction_scale="linear",
        )

        results = evaluator.evaluate(samples, batch_size=16, save_csv=False)

        self.assertEqual(len(results), 4 * len(KOZAK_CONTEXT_ORDER))

    def test_mutation_changes_only_codon_and_critical_context(self):
        original = np.zeros((60, 4), dtype=np.float32)
        original[:, 0] = 1.0
        mutated = mutate_cds_start_context(
            original, cds_start=12, start_codon="CTG", kozak_class="Weak"
        )
        changed = np.flatnonzero(np.any(original != mutated, axis=1)).tolist()

        self.assertEqual(changed, [9, 12, 13, 14, 15])
        self.assertEqual(np.argmax(mutated[9]), 1)
        self.assertEqual(
            [np.argmax(mutated[index]) for index in (12, 13, 14)],
            [1, 3, 2],
        )
        self.assertEqual(np.argmax(mutated[15]), 1)

    def test_collection_filters_cell_type_and_normalizes_enst_version(self):
        samples = collect_kozak_mutagenesis_samples(
            _make_dataset(),
            target_cell_type="brain",
            target_transcript_ids={"brain": ["ENST000001"]},
        )

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["tid"], "ENST000001")

    def test_evaluator_builds_complete_matched_design(self):
        samples = collect_kozak_mutagenesis_samples(_make_dataset())
        evaluator = KozakMutagenesisEvaluator(SequenceSensitiveModel())
        results = evaluator.evaluate(
            samples,
            batch_size=8,
            cds_skip_codons=1,
            save_csv=False,
        )

        self.assertEqual(len(results), 4 * len(KOZAK_CONTEXT_ORDER))
        self.assertEqual(results["Is_WT"].sum(), 1)
        np.testing.assert_allclose(
            results["Relative_Initiation_Efficiency"],
            1.0,
        )

    def test_plotting_writes_pdf_only(self):
        samples = collect_kozak_mutagenesis_samples(_make_dataset())
        evaluator = KozakMutagenesisEvaluator(SequenceSensitiveModel())
        results = evaluator.evaluate(samples, batch_size=16, save_csv=False)

        with TemporaryDirectory() as temporary_directory:
            paths = plot_kozak_mutagenesis_results(
                results, temporary_directory, suffix="test"
            )
            self.assertEqual(set(paths), {
                "boxplot", "per_codon_scatter", "global_scatter"
            })
            for path in paths.values():
                self.assertTrue(Path(path).is_file())
                self.assertEqual(Path(path).suffix, ".pdf")
            self.assertFalse(list(Path(temporary_directory).glob("*.png")))


if __name__ == "__main__":
    unittest.main()
