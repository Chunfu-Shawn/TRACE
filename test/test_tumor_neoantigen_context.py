#!/usr/bin/env python3
"""Regression tests for tumor neoantigen patient-context utilities."""

import os
import pathlib
import sys
import tempfile
import unittest

import pandas as pd


TOOLS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "tools", "tumor_neoantigen")
)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from filter_normal_proteome_offtargets import get_normal_proteome_path
from extract_specific_junctions import parse_gtf_junctions
from filter_gtex_step1 import assess_junction_background, clean_gtex_junction_id
from filter_gtex_step2 import permissive_background_pass
from metadata_utils import classify_tissue, find_patient_runs
from neoantigen_prioritization_report import build_patient_gtex_context, build_patient_jcpm_dict
from quantification_utils import calculate_gene_read_library_sizes, calculate_true_tpm
from cohort_annotation_utils import (
    antigen_origin_category,
    load_gtf_annotations,
    normalize_hla_a,
    population_coverage,
)
from select_shared_vaccine_peptides import largest_remainder_quotas, parse_netmhcpan_log
from run_trace_cohort_prediction import build_clean_sequence_dict, clean_id
from neoantigen_orf_config import build_neoantigen_orf_kwargs
from analyze_transcript_types import (
    MACRO_ORDER,
    MICRO_ORDER,
    annotate_transcripts,
    micro_category,
    summarize,
)
from analyze_tumor_associated_neoantigens import (
    add_sharing_statistics,
    classify_source,
    collapse_patient_peptides,
    peptide_crosses_junction,
)

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model.translation_utils import normalize_initiator_codon


class MetadataContextTests(unittest.TestCase):
    def test_de_novo_antigen_requires_same_transcript_sequence(self):
        row = {
            "Transcript_ID": "TX1",
            "Peptide": "PEPTIDE",
            "Is_De_Novo": True,
        }
        unverified = classify_source(
            row,
            canonical_proteins={},
            denovo_proteins={"TX2": "XXPEPTIDEXX"},
            class_codes={},
            exon_boundaries={},
        )
        verified = classify_source(
            row,
            canonical_proteins={},
            denovo_proteins={"TX1": "XXPEPTIDEXX"},
            class_codes={},
            exon_boundaries={},
        )
        self.assertEqual(unverified, ("De novo Gene", "Other ORFs", "id_only_not_sequence_verified"))
        self.assertEqual(verified, ("De novo Gene", "De novo Gene", "verified_id_and_sequence"))

    def test_peptide_junction_interval_uses_open_boundaries(self):
        boundaries = {"MSTRG.1": [9, 18]}
        self.assertTrue(peptide_crosses_junction("MSTRG.1", "6:12", boundaries))
        self.assertFalse(peptide_crosses_junction("MSTRG.1", "9:12", boundaries))
        self.assertFalse(peptide_crosses_junction("MSTRG.1", "Unmapped", boundaries))

    def test_patient_peptide_collapse_prevents_hla_double_counting(self):
        sources = pd.DataFrame(
            {
                "Patient": ["P1", "P1", "P1", "P2"],
                "Peptide": ["PEPTIDE"] * 4,
                "Macro_Origin": ["Canonical CDS", "Canonical CDS", "Novel Transcript", "Novel Transcript"],
                "Micro_Origin": ["Canonical CDS", "Canonical CDS", "Intergenic (u)", "Intergenic (u)"],
                "Clean_Transcript_ID": ["ENST1", "ENST1", "MSTRG.1", "MSTRG.1"],
                "MHC": ["HLA-A02:01", "HLA-A11:01", "HLA-A02:01", "HLA-A02:01"],
                "Aff(nM)": [30.0, 20.0, 50.0, 40.0],
                "%Rank_EL": [0.4, 0.2, 0.8, 0.5],
                "mean_intensity": [0.5, 0.5, 0.8, 0.7],
            }
        )
        peptide_units = collapse_patient_peptides(sources)
        peptide_units, sharing = add_sharing_statistics(peptide_units)
        self.assertEqual(len(peptide_units), 2)
        self.assertEqual(peptide_units.set_index("Patient").at["P1", "Macro_Origin"], "Multiple Origins")
        self.assertEqual(peptide_units.set_index("Patient").at["P1", "Best_Affinity_nM"], 20.0)
        self.assertEqual(sharing.iloc[0]["Shared_Patient_Count"], 2)

    def test_transcript_type_annotation_uses_denovo_and_novel_precedence(self):
        reference_gtf = (
            'chr1\ttest\ttranscript\t1\t100\t.\t+\t.\tgene_id "G1"; transcript_id "ENST1.1"; transcript_type "protein_coding";\n'
            'chr1\ttest\texon\t1\t100\t.\t+\t.\tgene_id "G1"; transcript_id "ENST1.1"; transcript_type "protein_coding";\n'
            'chr1\ttest\ttranscript\t201\t300\t.\t+\t.\tgene_id "G2"; transcript_id "ENST2.1"; transcript_type "processed_pseudogene";\n'
            'chr1\ttest\texon\t201\t300\t.\t+\t.\tgene_id "G2"; transcript_id "ENST2.1"; transcript_type "processed_pseudogene";\n'
        )
        novel_gtf = (
            'chr1\ttest\ttranscript\t401\t500\t.\t+\t.\tgene_id "MSTRG.1"; transcript_id "MSTRG.1.1";\n'
            'chr1\ttest\texon\t401\t500\t.\t+\t.\tgene_id "MSTRG.1"; transcript_id "MSTRG.1.1";\n'
        )
        with tempfile.NamedTemporaryFile("w", suffix=".gtf", delete=False) as ref_handle:
            ref_handle.write(reference_gtf)
            ref_path = ref_handle.name
        with tempfile.NamedTemporaryFile("w", suffix=".gtf", delete=False) as novel_handle:
            novel_handle.write(novel_gtf)
            novel_path = novel_handle.name
        frame = pd.DataFrame(
            {
                "Patient": ["P 1", "P 1", "P 1", "P 1"],
                "Transcript_ID": ["ENST1.1", "ENST2.1", "MSTRG.1.1", "MSTRG.1.1"],
                "Class_Code": ["Unknown", "Unknown", "u", "u"],
                "Biotype": ["old_annotation"] * 4,
            }
        )
        try:
            annotated, metrics = annotate_transcripts(
                frame,
                reference_gtf=ref_path,
                novel_gtf=novel_path,
                denovo_ids={"ENST1"},
            )
        finally:
            os.unlink(ref_path)
            os.unlink(novel_path)
        categories = annotated.set_index("Clean_Transcript_ID")["Broad_Category"].to_dict()
        self.assertEqual(categories["ENST1"], "De novo Gene")
        self.assertEqual(categories["ENST2"], "Pseudogene")
        self.assertEqual(categories["MSTRG.1.1"], "Novel Transcript")
        self.assertEqual(
            annotated.set_index("Clean_Transcript_ID").at["MSTRG.1.1", "Sub_Category"],
            "Intergenic (u)",
        )
        self.assertEqual(metrics["Duplicate_Rows_Removed"], 1)

    def test_transcript_type_summary_retains_zero_count_patients(self):
        frame = pd.DataFrame(
            {
                "Patient": ["P1", "P2"],
                "Broad_Category": ["Protein Coding", "Novel Transcript"],
            }
        )
        matrix = summarize(frame, "Broad_Category", MACRO_ORDER, ["P1", "P2", "P3"])
        self.assertEqual(matrix.loc["P1", "Protein Coding"], 1)
        self.assertEqual(matrix.loc["P3"].sum(), 0)
        self.assertEqual(
            micro_category({"Broad_Category": "Novel Transcript", "Class_Code": "m"}),
            "Retained Intron (m/n)",
        )

    def test_transcript_type_summary_handles_no_novel_transcripts(self):
        matrix = summarize(
            pd.DataFrame(columns=["Patient", "Sub_Category"]),
            "Sub_Category",
            MICRO_ORDER,
            ["P1", "P2"],
        )
        self.assertEqual(matrix.index.tolist(), ["P1", "P2"])
        self.assertEqual(matrix.shape, (2, 0))

    def test_neoantigen_orf_configuration_matches_coding_orf_profile(self):
        config = build_neoantigen_orf_kwargs()
        self.assertEqual(config["start_codons"], ["ATG", "CTG", "GTG", "TTG"])
        self.assertEqual(config["mode"], "balanced")
        self.assertFalse(config["long_mode_length_only"])
        self.assertEqual(config["hard_thresh_intensity"], 0.01)
        self.assertEqual(config["hard_thresh_periodicity"], 0.5)
        self.assertEqual(config["hard_thresh_uniformity"], 0.8)
        self.assertEqual(config["hard_thresh_step_up"], 0.5)
        self.assertEqual(config["hard_thresh_drop_off"], 0.8)
        self.assertEqual(config["ranking_strategy"], "occupancy_expression")
        self.assertEqual(config["score_features"], ["step_up_contrast", "drop_off"])
        self.assertEqual(config["tpm_exponent"], 1.0)
        self.assertEqual(config["collapse_boundary_weight"], 0.5)
        self.assertEqual(config["start_codon_prior_strength"], 0.25)
        self.assertEqual(config["nms_iou_threshold"], 0.7)
        self.assertFalse(config["nms_respect_frame"])

    def test_common_normal_labels_are_not_classified_as_tumor(self):
        normal_labels = [
            "normal",
            "non-tumor",
            "non_tumour tissue",
            "adjacent",
            "benign",
            "healthy control",
            "paratumor",
        ]
        for label in normal_labels:
            with self.subTest(label=label):
                self.assertEqual(classify_tissue(label), "normal")

        self.assertEqual(classify_tissue("primary tumor"), "tumor")
        self.assertEqual(classify_tissue("malignant carcinoma"), "tumor")

    def test_patient_run_lookup_uses_normalized_labels(self):
        metadata = (
            "Run,Individual,tissue\n"
            "T_RUN,patient 1,Tumor\n"
            "N_RUN,patient 1,non-tumor\n"
            "OTHER,patient 10,normal\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as handle:
            handle.write(metadata)
            path = handle.name
        try:
            self.assertEqual(find_patient_runs(path, "patient 1", "tumor"), ["T_RUN"])
            self.assertEqual(find_patient_runs(path, "patient 1", "normal"), ["N_RUN"])
        finally:
            os.unlink(path)

    def test_junction_lookup_is_limited_to_one_tumor_run(self):
        step2 = pd.DataFrame(
            {
                "Transcript_ID": ["MSTRG.1", "MSTRG.1", "MSTRG.2"],
                "Tumor_Run": ["RUN_A", "RUN_B", "RUN_A"],
                "Tumor_Junction_CPM": [2.0, 99.0, 4.0],
            }
        )
        lookup = build_patient_jcpm_dict(step2, "RUN_A")
        self.assertEqual(lookup, {"MSTRG.1": 2.0, "MSTRG.2": 4.0})

    def test_gtex_assessment_context_is_limited_to_one_tumor_run(self):
        step2 = pd.DataFrame(
            {
                "Transcript_ID": ["MSTRG.1", "MSTRG.1"],
                "Tumor_Run": ["RUN_A", "RUN_B"],
                "Tumor_Junction_CPM": [2.0, 99.0],
                "GTEx_Transcript_TPM_Covered": [False, True],
                "GTEx_Transcript_Filter_Source": [
                    "Not_assessed_no_precomputed_transcript",
                    "Complete_GTF_GTEx_Step2",
                ],
                "GTEx_Step2_Applied": [False, True],
            }
        )
        context = build_patient_gtex_context(step2, "RUN_A")
        self.assertFalse(context["MSTRG.1"]["GTEx_Transcript_TPM_Covered"])
        self.assertEqual(
            context["MSTRG.1"]["GTEx_Transcript_Filter_Source"],
            "Not_assessed_no_precomputed_transcript",
        )

    def test_normal_proteome_path_contains_normal_subdirectory(self):
        path = get_normal_proteome_path("/trace", "patient_1", "short")
        self.assertEqual(
            path,
            "/trace/patient_1/normal/high_confidence_proteins.patient_1.short_mode.fasta",
        )

    def test_true_tpm_uses_total_transcript_rpk(self):
        counts = pd.DataFrame({'RUN_A': [100, 100]}, index=['TX1', 'TX2'])
        lengths = pd.Series([1000, 2000], index=['TX1', 'TX2'])
        tpm = calculate_true_tpm(counts, lengths)
        self.assertAlmostEqual(tpm['RUN_A'].sum(), 1_000_000.0)
        self.assertAlmostEqual(tpm.at['TX1', 'RUN_A'], 666_666.666666, places=3)
        self.assertAlmostEqual(tpm.at['TX2', 'RUN_A'], 333_333.333333, places=3)

    def test_gene_read_library_size_is_column_sum(self):
        counts_text = (
            "Geneid\tChr\tStart\tEnd\tStrand\tLength\t/path/RUN_A.uniq.sorted.bam\n"
            "G1\tchr1\t1\t100\t+\t100\t10\n"
            "G2\tchr1\t200\t300\t+\t101\t15\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
            handle.write(counts_text)
            path = handle.name
        try:
            library_sizes = calculate_gene_read_library_sizes(path)
            self.assertEqual(library_sizes.to_dict(), {'RUN_A': 25})
        finally:
            os.unlink(path)

    def test_junction_ids_preserve_strand(self):
        gtf = (
            'chr1\ttest\texon\t100\t150\t.\t+\t.\tgene_id "G1"; transcript_id "TX1";\n'
            'chr1\ttest\texon\t200\t250\t.\t+\t.\tgene_id "G1"; transcript_id "TX1";\n'
        )
        with tempfile.NamedTemporaryFile("w", suffix=".gtf", delete=False) as handle:
            handle.write(gtf)
            path = handle.name
        try:
            junction_map, _ = parse_gtf_junctions(path)
            self.assertIn('chr1:150-200:+', junction_map)
            self.assertEqual(clean_gtex_junction_id('chr1:150-200:+'), 'chr1:150-200:+')
        finally:
            os.unlink(path)

    def test_missing_junction_background_is_marked_and_passable(self):
        assessment = assess_junction_background(
            ['chr1:150-200:+'],
            {},
        )
        self.assertFalse(assessment['GTEx_Junction_Background_Assessed'])
        self.assertEqual(assessment['GTEx_Junction_Coverage'], 'Not_found')
        self.assertEqual(
            assessment['GTEx_Junction_IDs_Missing'],
            'chr1:150-200:+',
        )
        self.assertTrue(pd.isna(assessment['Global_Max_GTEx_JCPM']))

        nan_assessment = assess_junction_background(
            ['chr1:150-200:+'],
            {'chr1:150-200:+': float('nan')},
        )
        self.assertFalse(nan_assessment['GTEx_Junction_Background_Assessed'])

    def test_partial_junction_background_keeps_missing_id_marker(self):
        assessment = assess_junction_background(
            ['chr1:150-200:+', 'chr2:250-300:-'],
            {'chr1:150-200:+': 0.05},
        )
        self.assertTrue(assessment['GTEx_Junction_Background_Assessed'])
        self.assertEqual(assessment['GTEx_Junction_Coverage'], 'Partial')
        self.assertEqual(assessment['Global_Max_GTEx_JCPM'], 0.05)
        self.assertEqual(
            assessment['GTEx_Junction_IDs_Missing'],
            'chr2:250-300:-',
        )

    def test_step2_missing_requantified_transcript_is_passed(self):
        covered = pd.Series([False, True, True])
        values = pd.Series([float('nan'), 0.1, 3.0])
        passed = permissive_background_pass(covered, values, threshold=0.5)
        self.assertEqual(passed.tolist(), [True, True, False])

    def test_near_cognate_initiator_is_translated_as_methionine(self):
        self.assertEqual(normalize_initiator_codon('CTGAAATAA'), 'ATGAAATAA')
        self.assertEqual(normalize_initiator_codon('GTGAAATAA'), 'ATGAAATAA')
        self.assertEqual(normalize_initiator_codon('TTGAAATAA'), 'ATGAAATAA')
        self.assertEqual(normalize_initiator_codon('ACGAAATAA'), 'ATGAAATAA')
        self.assertEqual(normalize_initiator_codon('ATGCTGTAA'), 'ATGCTGTAA')

    def test_gtf_cds_coordinates_are_transcript_relative_on_both_strands(self):
        gtf = (
            'chr1\ttest\ttranscript\t100\t209\t.\t+\t.\tgene_id "G1"; transcript_id "TXP"; transcript_type "protein_coding";\n'
            'chr1\ttest\texon\t100\t109\t.\t+\t.\tgene_id "G1"; transcript_id "TXP";\n'
            'chr1\ttest\texon\t200\t209\t.\t+\t.\tgene_id "G1"; transcript_id "TXP";\n'
            'chr1\ttest\tstart_codon\t100\t102\t.\t+\t0\tgene_id "G1"; transcript_id "TXP";\n'
            'chr1\ttest\tstop_codon\t207\t209\t.\t+\t0\tgene_id "G1"; transcript_id "TXP";\n'
            'chr1\ttest\ttranscript\t100\t209\t.\t-\t.\tgene_id "G2"; transcript_id "TXM"; transcript_type "protein_coding";\n'
            'chr1\ttest\texon\t100\t109\t.\t-\t.\tgene_id "G2"; transcript_id "TXM";\n'
            'chr1\ttest\texon\t200\t209\t.\t-\t.\tgene_id "G2"; transcript_id "TXM";\n'
            'chr1\ttest\tstart_codon\t207\t209\t.\t-\t0\tgene_id "G2"; transcript_id "TXM";\n'
            'chr1\ttest\tstop_codon\t100\t102\t.\t-\t0\tgene_id "G2"; transcript_id "TXM";\n'
        )
        with tempfile.NamedTemporaryFile("w", suffix=".gtf", delete=False) as handle:
            handle.write(gtf)
            path = handle.name
        try:
            annotation = load_gtf_annotations(path, {"TXP", "TXM"}).set_index("Transcript_ID")
            self.assertEqual(annotation.at["TXP", "Canonical_ORF_Start"], 0)
            self.assertEqual(annotation.at["TXP", "Canonical_ORF_Stop"], 17)
            self.assertEqual(annotation.at["TXM", "Canonical_ORF_Start"], 0)
            self.assertEqual(annotation.at["TXM", "Canonical_ORF_Stop"], 17)
        finally:
            os.unlink(path)

    def test_antigen_origin_keeps_unresolved_protein_coding_explicit(self):
        canonical = {
            "Transcript_ID": "ENST1",
            "Biotype": "protein_coding",
            "Canonical_ORF_Start": 0,
            "Canonical_ORF_Stop": 99,
            "ORF_Pos": "3:96",
        }
        unresolved = {
            "Transcript_ID": "ENST2",
            "Biotype": "protein_coding",
            "Canonical_ORF_Start": pd.NA,
            "Canonical_ORF_Stop": pd.NA,
            "ORF_Pos": "0:99",
        }
        self.assertEqual(antigen_origin_category(canonical, tolerance=6), "Canonical CDS")
        self.assertEqual(
            antigen_origin_category(unresolved),
            "Unresolved protein-coding ORF",
        )

    def test_hla_frequency_math_and_panel_quotas(self):
        self.assertEqual(normalize_hla_a("HLA-A02:01"), "HLA-A*02:01")
        self.assertEqual(normalize_hla_a("A*11:01"), "HLA-A*11:01")
        self.assertAlmostEqual(population_coverage(0.20), 0.36)
        quotas = largest_remainder_quotas(
            {"HLA-A*11:01": 0.20, "HLA-A*24:02": 0.15, "HLA-A*02:01": 0.10},
            20,
        )
        self.assertEqual(sum(quotas.values()), 20)
        self.assertGreater(quotas["HLA-A*11:01"], quotas["HLA-A*02:01"])

    def test_pan_hla_raw_log_parser(self):
        raw_line = (
            "1 HLA-A02:01 PEPTIDEAA PEPTIDEA 0 0 0 0 0 0 TARGET "
            "0.5 1.0 0.2 0.8 100.0 SB\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as handle:
            handle.write(raw_line)
            path = handle.name
        try:
            parsed = parse_netmhcpan_log(pathlib.Path(path))
            self.assertEqual(parsed.iloc[0]["HLA"], "HLA-A*02:01")
            self.assertEqual(parsed.iloc[0]["Peptide"], "PEPTIDEAA")
            self.assertEqual(parsed.iloc[0]["Rank_EL"], 1.0)
        finally:
            os.unlink(path)

    def test_cohort_fasta_keys_keep_enst_and_novel_transcripts(self):
        sequences = {
            "ENST000001.7|ENSG000001.3": "AUGAAATAA",
            "MSTRG.12.3": "CTGAAATAA",
        }
        cleaned = build_clean_sequence_dict(sequences)
        self.assertEqual(clean_id("ENST000001.7|ENSG000001.3"), "ENST000001")
        self.assertEqual(clean_id("MSTRG.12.3"), "MSTRG.12.3")
        self.assertEqual(cleaned["ENST000001"], "ATGAAATAA")
        self.assertEqual(cleaned["MSTRG.12.3"], "CTGAAATAA")

    def test_cohort_fasta_rejects_conflicting_normalized_enst_versions(self):
        with self.assertRaises(ValueError):
            build_clean_sequence_dict(
                {
                    "ENST000001.1|ENSG000001.1": "ATGAAATAA",
                    "ENST000001.2|ENSG000001.1": "ATGCCCTAA",
                }
            )


if __name__ == "__main__":
    unittest.main()
