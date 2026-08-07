#!/home/user/data2/rbase/envs/anaconda3/envs/ribo_model/bin/python
import os
import random
import math
import argparse
import numpy as np
import torch
import pandas as pd
from typing import Any, Callable, Tuple, List, Optional, Dict
import sys
import matplotlib.pyplot as plt

# Model Loading and Environment Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
from model.prediction_heads import PsiteDensityHead
from model.base_model import BaseModel
from model.prediction_heads import TranslationProfileHead

def tokenize_seq_onehot(seq_str, d_seq=4):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3}
    one_hot_list = []
    for base in seq_str.upper():
        idx = mapping.get(base, -1)
        vec = [0.0] * d_seq
        if idx != -1: vec[idx] = 1.0
        one_hot_list.append(vec)
    return torch.tensor(one_hot_list, dtype=torch.float32).unsqueeze(0)


# Batched Beam Search Optimizer with Cell-Type Specificity Constraints
class BatchedBeamOptimizer:
    STANDARD_GENETIC_CODE = {
        'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M', 'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
        'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K', 'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
        'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L', 'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
        'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q', 'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
        'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V', 'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
        'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E', 'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
        'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S', 'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
        'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*', 'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W',
    }

    def __init__(self, model, tokenizer_fn: Callable[[str], torch.Tensor], 
                 target_cell_type: str="HEK293T_inhouse", 
                 negative_cell_types: Optional[List[str]] = None,
                 species: str="human", 
                 expr_dict_path: Optional[str] = None, 
                 head_name: str="count", seed: int=42):
        self.model = model
        self.tokenizer_fn = tokenizer_fn
        self.target_cell_type = target_cell_type
        self.negative_cell_types = negative_cell_types if negative_cell_types is not None else []
        self.valid_negative_cell_types = []
        self.species = species
        self.head_name = head_name
        self.bases = ['A', 'T', 'C', 'G']
        self.aa_to_codons = self._build_codon_table()
        
        self.expr_vectors = {}
        if expr_dict_path is not None:
            expr_dict = torch.load(expr_dict_path, map_location='cpu')
            
            # Load Target Cell Type Expression Vector
            if self.target_cell_type in expr_dict:
                self.expr_vectors[self.target_cell_type] = expr_dict[self.target_cell_type]
                print(f"Loaded target vector: {self.target_cell_type}")
            else:
                print(f"Warning: Target {self.target_cell_type} not found. Operating without specific target vector.")
            
            # Load Negative Cell Type Expression Vectors
            for neg_cell in self.negative_cell_types:
                if neg_cell in expr_dict:
                    self.expr_vectors[neg_cell] = expr_dict[neg_cell]
                    self.valid_negative_cell_types.append(neg_cell)
                    print(f"Loaded negative vector: {neg_cell}")
                else:
                    print(f"Warning: Negative target {neg_cell} not found. It will be ignored.")
        else:
            print(f"Warning: expr_dict_path not provided. Using hardcoded cell types.")

        self.seed = seed
        self.set_seed(seed)

    def set_seed(self, seed: int):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _build_codon_table(self):
        aa_to_codons = {}
        for codon, aa in self.STANDARD_GENETIC_CODE.items():
            if aa not in aa_to_codons: aa_to_codons[aa] = []
            aa_to_codons[aa].append(codon)
        return aa_to_codons

    @staticmethod
    def _to_linear_signal(raw_signal: np.ndarray) -> np.ndarray:
        """Convert the model's log1p-scale output back to linear density."""
        raw_signal = np.asarray(raw_signal, dtype=np.float64)
        linear_signal = np.expm1(raw_signal)
        linear_signal = np.nan_to_num(
            linear_signal, nan=0.0, posinf=np.finfo(np.float64).max, neginf=0.0
        )
        return np.maximum(linear_signal, 0.0)

    @torch.no_grad()
    def predict_signal(self, seq: str, cell_type: Optional[str] = None) -> np.ndarray:
        """Predict a full-length, one-dimensional P-site density profile in linear space."""
        query_cell = cell_type if cell_type else self.target_cell_type
        seq_tensor = self.tokenizer_fn(seq)
        outputs = self.model.predict(
            seq_batch=seq_tensor, cell_type=query_cell,
            expr_vector=self.expr_vectors.get(query_cell), species=self.species,
            head_names=[self.head_name], return_numpy=True
        )
        return self._to_linear_signal(outputs[self.head_name]).reshape(-1)

    @torch.no_grad()
    def get_signal_profile(self, seq: str, cds_start: int, cds_end: int) -> np.ndarray:
        signals = self.predict_signal(seq, self.target_cell_type)
        valid_end = min(len(signals), cds_end)
        return signals[cds_start:valid_end:3]

    @torch.no_grad()
    def predict_te(self, seq: str, cds_start: int, cds_end: int, cell_type: str = None) -> float:
        query_cell = cell_type if cell_type else self.target_cell_type
        signals = self.predict_signal(seq, query_cell)
        seq_len = len(signals)
        if cds_start >= seq_len or cds_start >= min(seq_len, cds_end): return 0.0
        valid_end = min(seq_len, cds_end)
        return float(np.mean(signals[cds_start:valid_end:3]))

    @torch.no_grad()
    def evaluate_batch(self, seq_list: List[str], cds_start: int, cds_end: int,
                    wt_profile: Optional[np.ndarray] = None, 
                    drop_tolerance: float = 0.5, penalty_weight: float = 0.2,
                    consistency_weight: float = 0.25, ratio_weight: float = 0.05,
                    utr5_penalty_weight: float = 0.1, aug_context_weight: float = 0.1,
                    specificity_weight: float = 1.0, consistency_tolerance: float = 0.1,
                    utr5_tolerance: float = 0.1, aug_context_tolerance: float = 0.1
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        tensor_list = [self.tokenizer_fn(s) for s in seq_list]
        batch_tensor = torch.cat(tensor_list, dim=0) 

        # 1. Target Cell TE
        outputs_target = self.model.predict(
            seq_batch=batch_tensor, cell_type=self.target_cell_type,
            expr_vector=self.expr_vectors.get(self.target_cell_type), species=self.species,
            head_names=[self.head_name], return_numpy=True
        )
        signals_flat = self._to_linear_signal(outputs_target[self.head_name]).reshape(len(seq_list), -1)
        seq_len = signals_flat.shape[1]
        valid_end = min(seq_len, cds_end)
        morf_signals = signals_flat[:, cds_start:valid_end:3]
        morf_mean = np.mean(morf_signals, axis=1)

        # 2. Negative Cells TE (Max-Penalty Strategy)
        max_neg_te = np.zeros(len(seq_list))
        if self.valid_negative_cell_types and specificity_weight > 0:
            for neg_cell in self.valid_negative_cell_types:
                outputs_neg = self.model.predict(
                    seq_batch=batch_tensor, cell_type=neg_cell,
                    expr_vector=self.expr_vectors.get(neg_cell), species=self.species,
                    head_names=[self.head_name], return_numpy=True
                )
                neg_signals_flat = self._to_linear_signal(outputs_neg[self.head_name]).reshape(len(seq_list), -1)
                neg_morf_signals = neg_signals_flat[:, cds_start:valid_end:3]
                neg_morf_mean = np.mean(neg_morf_signals, axis=1)
                max_neg_te = np.maximum(max_neg_te, neg_morf_mean)

        # 3. Biological Penalties
        eps = 1e-6
        morf_std = np.std(morf_signals, axis=1)
        cv = morf_std / (morf_mean + eps)
        cv_scale = max(self.wt_cv, 0.1)
        relative_cv_increase = (cv - self.wt_cv) / cv_scale
        consistency_penalty = (
            np.maximum(0.0, relative_cv_increase - consistency_tolerance)
            * consistency_weight
        )

        utr5_absolute_penalty = np.zeros(len(seq_list))
        if cds_start > 0:
            utr5_signals = signals_flat[:, :cds_start]
            utr5_mean_raw = np.mean(utr5_signals, axis=1)
            utr_scale = max(self.wt_utr5_mean, self.wt_target_te * 0.05, eps)
            relative_utr_increase = (utr5_mean_raw - self.wt_utr5_mean) / utr_scale
            utr5_absolute_penalty = (
                np.maximum(0.0, relative_utr_increase - utr5_tolerance)
                * utr5_penalty_weight
            )
            utr5_mean_safe = np.maximum(utr5_mean_raw, eps)
            cds_utr_ratio = morf_mean / utr5_mean_safe
        else:
            cds_utr_ratio = np.zeros(len(seq_list))
        ratio_scale = max(self.wt_cds_utr_ratio, eps)
        relative_ratio_improvement = np.maximum(0.0, cds_utr_ratio / ratio_scale - 1.0)
        ratio_bonus = np.clip(relative_ratio_improvement, 0.0, 2.0) * ratio_weight

        wt_penalty = np.zeros(len(seq_list))
        if wt_profile is not None:
            min_len = min(len(wt_profile), morf_signals.shape[1])
            wt_reference = wt_profile[:min_len]
            allowed_floor = wt_reference * (1.0 - drop_tolerance)
            violations = np.maximum(0.0, allowed_floor - morf_signals[:, :min_len])
            relative_violations = violations / np.maximum(wt_reference, eps)
            wt_penalty = np.mean(relative_violations, axis=1) * penalty_weight

        aug_context_end = min(seq_len, cds_start + 30)
        if aug_context_end > cds_start:
            aug_signals = signals_flat[:, cds_start:aug_context_end:3]
            aug_mean = np.mean(aug_signals, axis=1)
            aug_ratio = aug_mean / (morf_mean + eps)
            aug_scale = max(self.wt_aug_context_ratio, 0.1)
            relative_aug_increase = (aug_ratio - self.wt_aug_context_ratio) / aug_scale
            aug_context_penalty = (
                np.maximum(0.0, relative_aug_increase - aug_context_tolerance)
                * aug_context_weight
            )
        else:
            aug_context_penalty = np.zeros(len(seq_list))

        # 4. Fold-Change Fitness & Integration
        target_fc = morf_mean / (self.wt_target_te + eps)
        log_target_fc = np.log2(np.maximum(target_fc, eps))

        # Graceful degradation logic for absence of negative cells
        if self.valid_negative_cell_types and specificity_weight > 0:
            neg_fc = max_neg_te / (self.wt_max_neg_te + eps)

            log_neg_fc = np.log2(np.maximum(neg_fc, eps))
            
            log_specificity = log_target_fc - log_neg_fc
            specificity_fc = np.exp2(log_specificity)
            
            fitness_score = log_target_fc - specificity_weight * log_neg_fc
        else:
            neg_fc = np.ones(len(seq_list))
            specificity_fc = target_fc.copy()
            fitness_score = log_target_fc

        # Apply biological penalties and bonuses
        fitness_score = (
            fitness_score
            + ratio_bonus
            - wt_penalty
            - consistency_penalty
            - utr5_absolute_penalty
            - aug_context_penalty
        )

        return (
            fitness_score,
            morf_mean,
            max_neg_te,
            specificity_fc
        )

    @staticmethod
    def _bounded_mutation_rate(
        unit_count: int, base_rate: float,
        min_expected_mutations: float, max_expected_mutations: float
    ) -> Tuple[float, float]:
        """Return a length-adaptive per-unit rate and its bounded expected burden."""
        if unit_count <= 0 or base_rate <= 0.0:
            return 0.0, 0.0
        if not 0.0 <= base_rate <= 1.0:
            raise ValueError("Base mutation rates must be between 0 and 1.")
        if min_expected_mutations < 0.0 or max_expected_mutations < min_expected_mutations:
            raise ValueError("Mutation burden bounds must satisfy 0 <= minimum <= maximum.")

        expected_mutations = float(np.clip(
            base_rate * unit_count,
            min_expected_mutations,
            max_expected_mutations,
        ))
        effective_rate = min(1.0, expected_mutations / unit_count)
        return effective_rate, effective_rate * unit_count

    @staticmethod
    def _normalized_operator_weights(
        mode: int, cds_fraction: float, utr5_fraction: float,
        utr3_fraction: float, joint_fraction: float,
        utr_only_utr5_fraction: float = 0.60,
        utr_only_utr3_fraction: float = 0.25,
        utr_only_joint_fraction: float = 0.15
    ) -> Dict[str, float]:
        """Build normalized mutation-operator weights for the selected optimization mode."""
        if mode == 1:
            weights = {
                "cds": cds_fraction,
                "utr5": utr5_fraction,
                "utr3": utr3_fraction,
                "joint": joint_fraction,
            }
        elif mode == 2:
            weights = {"cds": 1.0}
        elif mode == 3:
            weights = {"utr5": 1.0}
        elif mode == 4:
            weights = {"utr3": 1.0}
        elif mode == 5:
            weights = {
                "utr5": utr_only_utr5_fraction,
                "utr3": utr_only_utr3_fraction,
                "utr_joint": utr_only_joint_fraction,
            }
        else:
            raise ValueError(f"Unsupported optimization mode: {mode}")

        if any(weight < 0.0 for weight in weights.values()):
            raise ValueError("Mutation-operator fractions must be non-negative.")
        positive_weights = {name: weight for name, weight in weights.items() if weight > 0.0}
        total_weight = sum(positive_weights.values())
        if total_weight <= 0.0:
            raise ValueError("At least one mutation-operator fraction must be positive.")
        return {name: weight / total_weight for name, weight in positive_weights.items()}

    @staticmethod
    def _allocate_operator_counts(total_count: int, weights: Dict[str, float]) -> Dict[str, int]:
        """Allocate an exact integer candidate quota using the largest-remainder method."""
        if total_count <= 0:
            return {name: 0 for name in weights}
        raw_counts = {name: total_count * weight for name, weight in weights.items()}
        counts = {name: int(math.floor(value)) for name, value in raw_counts.items()}
        remainder = total_count - sum(counts.values())
        ranked = sorted(
            weights,
            key=lambda name: (raw_counts[name] - counts[name], weights[name], name),
            reverse=True,
        )
        for name in ranked[:remainder]:
            counts[name] += 1
        return counts

    def _mutate_sequence(
        self, seq: str, cds_start: int, cds_end: int, operator: str,
        utr5_mutation_rate: float, utr3_mutation_rate: float,
        cds_mutation_rate: float
    ) -> str:
        """Apply one region-specific mutation operator with independent effective rates."""
        operator_regions = {
            "cds": (True, False, False),
            "utr5": (False, True, False),
            "utr3": (False, False, True),
            "utr_joint": (False, True, True),
            "joint": (True, True, True),
        }
        if operator not in operator_regions:
            raise ValueError(f"Unsupported mutation operator: {operator}")
        if any(not 0.0 <= rate <= 1.0 for rate in (
            utr5_mutation_rate, utr3_mutation_rate, cds_mutation_rate,
        )):
            raise ValueError("Effective mutation rates must be between 0 and 1.")

        mutate_cds, mutate_utr5, mutate_utr3 = operator_regions[operator]
        seq_list = list(seq.upper().replace('U', 'T'))

        if mutate_cds:
            complete_cds_end = cds_start + ((cds_end - cds_start) // 3) * 3
            for codon_start in range(cds_start, complete_cds_end, 3):
                if random.random() > cds_mutation_rate:
                    continue
                old_codon = "".join(seq_list[codon_start:codon_start + 3])
                aa = self.STANDARD_GENETIC_CODE.get(old_codon)
                if aa is None:
                    continue
                synonymous_codons = [c for c in self.aa_to_codons[aa] if c != old_codon]
                if synonymous_codons:
                    seq_list[codon_start:codon_start + 3] = list(random.choice(synonymous_codons))

        if mutate_utr5:
            utr5_end = max(0, min(cds_start, len(seq_list)))
            for pos in range(utr5_end):
                if random.random() > utr5_mutation_rate:
                    continue
                old_base = seq_list[pos]
                new_base = random.choice([base for base in self.bases if base != old_base])
                temp_seq = seq_list.copy()
                temp_seq[pos] = new_base
                local_start = max(0, pos - 2)
                local_end = min(cds_start, pos + 3)
                if "ATG" not in "".join(temp_seq[local_start:local_end]):
                    seq_list[pos] = new_base

        if mutate_utr3:
            for pos in range(max(0, cds_end), len(seq_list)):
                if random.random() > utr3_mutation_rate:
                    continue
                old_base = seq_list[pos]
                seq_list[pos] = random.choice([base for base in self.bases if base != old_base])

        return "".join(seq_list)

    @staticmethod
    def _edit_distance_bin(total_edits: int) -> str:
        """Return a compact label for edit-distance-stratified candidate archives."""
        if total_edits <= 2:
            return "0-2"
        if total_edits <= 5:
            return "3-5"
        if total_edits <= 10:
            return "6-10"
        if total_edits <= 20:
            return "11-20"
        return "21+"

    @classmethod
    def _mutation_load(
        cls, seq: str, wt_seq: str, cds_start: int, cds_end: int,
        cds_mutable_codon_count: int, active_unit_count: int,
    ) -> Dict[str, Any]:
        """Measure cumulative edits relative to WT using nucleotides for UTRs and codons for CDS."""
        utr5_edits = sum(a != b for a, b in zip(seq[:cds_start], wt_seq[:cds_start]))
        utr3_edits = sum(a != b for a, b in zip(seq[cds_end:], wt_seq[cds_end:]))
        cds_codon_edits = sum(
            seq[pos:pos + 3] != wt_seq[pos:pos + 3]
            for pos in range(cds_start, cds_end, 3)
        )
        cds_nt_edits = sum(
            a != b for a, b in zip(seq[cds_start:cds_end], wt_seq[cds_start:cds_end])
        )
        total_design_edits = utr5_edits + cds_codon_edits + utr3_edits

        utr5_units = cds_start
        utr3_units = len(wt_seq) - cds_end
        return {
            "utr5_edits": int(utr5_edits),
            "cds_codon_edits": int(cds_codon_edits),
            "cds_nt_edits": int(cds_nt_edits),
            "utr3_edits": int(utr3_edits),
            "total_design_edits": int(total_design_edits),
            "utr5_edit_fraction": utr5_edits / max(utr5_units, 1),
            "cds_edit_fraction": cds_codon_edits / max(cds_mutable_codon_count, 1),
            "utr3_edit_fraction": utr3_edits / max(utr3_units, 1),
            "total_mutation_burden": total_design_edits / max(active_unit_count, 1),
            "edit_distance_bin": cls._edit_distance_bin(total_design_edits),
        }

    @staticmethod
    def _pareto_front_indices(records: List[Dict[str, Any]]) -> List[int]:
        """Return non-dominated indices for minimizing burden and maximizing adjusted gain."""
        ranked_indices = sorted(
            range(len(records)),
            key=lambda idx: (
                records[idx]["total_mutation_burden"],
                -records[idx]["prior_adjusted_gain"],
                -records[idx]["target_te"],
                records[idx]["sequence"],
            ),
        )
        front = []
        best_gain = -np.inf
        for idx in ranked_indices:
            gain = records[idx]["prior_adjusted_gain"]
            if gain > best_gain + 1e-12:
                front.append(idx)
                best_gain = gain
        return front

    @staticmethod
    def _select_pareto_knee(
        records: List[Dict[str, Any]], front_indices: List[int]
    ) -> int:
        """Select the maximum-curvature point on a normalized two-objective Pareto front."""
        if not front_indices:
            raise ValueError("Cannot select a Pareto knee from an empty front.")
        if len(front_indices) == 1:
            return front_indices[0]

        ordered = sorted(
            front_indices,
            key=lambda idx: (
                records[idx]["total_mutation_burden"],
                records[idx]["prior_adjusted_gain"],
            ),
        )
        burdens = np.array(
            [records[idx]["total_mutation_burden"] for idx in ordered], dtype=float
        )
        gains = np.array([records[idx]["prior_adjusted_gain"] for idx in ordered], dtype=float)

        burden_span = float(np.ptp(burdens))
        gain_span = float(np.ptp(gains))
        if len(ordered) < 3 or burden_span <= 1e-12 or gain_span <= 1e-12:
            return max(
                ordered,
                key=lambda idx: (
                    records[idx]["prior_adjusted_gain"]
                    / max(records[idx]["total_mutation_burden"], 1e-12),
                    records[idx]["prior_adjusted_gain"],
                    -records[idx]["total_mutation_burden"],
                ),
            )

        x = (burdens - burdens.min()) / burden_span
        y = (gains - gains.min()) / gain_span
        start = np.array([x[0], y[0]])
        end = np.array([x[-1], y[-1]])
        chord = end - start
        chord_norm = float(np.linalg.norm(chord))
        points = np.column_stack((x, y))
        offsets = points - start
        distances = np.abs(
            chord[0] * offsets[:, 1] - chord[1] * offsets[:, 0]
        ) / chord_norm
        max_distance = float(np.max(distances))
        knee_positions = np.flatnonzero(np.isclose(distances, max_distance, atol=1e-12))
        return max(
            (ordered[int(pos)] for pos in knee_positions),
            key=lambda idx: (
                records[idx]["prior_adjusted_gain"],
                -records[idx]["total_mutation_burden"],
            ),
        )

    def optimize(
        self, full_seq: str, cds_start: int, cds_end: int, 
        mode: int = 1, iterations: int = 200, batch_size: int = 128, 
        mutation_rate: Optional[float] = None,
        cds_mutation_rate: Optional[float] = None,
        utr5_base_rate: float = 0.008, utr3_base_rate: float = 0.002,
        cds_base_rate: float = 0.01,
        utr5_min_expected_mutations: float = 0.5,
        utr5_max_expected_mutations: float = 2.0,
        utr3_min_expected_mutations: float = 0.5,
        utr3_max_expected_mutations: float = 2.0,
        cds_min_expected_mutations: float = 1.0,
        cds_max_expected_mutations: float = 3.0,
        global_cds_fraction: float = 0.40,
        global_utr5_fraction: float = 0.25,
        global_utr3_fraction: float = 0.15,
        global_joint_fraction: float = 0.20,
        utr_only_utr5_fraction: float = 0.60,
        utr_only_utr3_fraction: float = 0.25,
        utr_only_joint_fraction: float = 0.15,
        beam_width: int = 8, 
        use_continuity_constraint: bool = True, drop_tolerance: float = 0.5,        
        penalty_weight: float = 0.2, consistency_weight: float = 0.25,    
        ratio_weight: float = 0.05, utr5_penalty_weight: float = 0.1,
        aug_context_weight: float = 0.1, specificity_weight: float = 1.0,
        consistency_tolerance: float = 0.1, utr5_tolerance: float = 0.1,
        aug_context_tolerance: float = 0.1, min_gain: float = 0.01,
        patience: int = 50, max_candidate_attempts: Optional[int] = None,
        max_lineage_depth: int = 10,
        max_utr5_mutation_fraction: float = 0.10,
        max_cds_mutation_fraction: float = 0.10,
        max_utr3_mutation_fraction: float = 0.10,
        max_total_mutation_fraction: float = 0.10,
        high_te_exclusion_fraction: float = 0.20,
    ) -> Tuple[str, float, List[Dict], List[float], List[float]]:
        
        self.set_seed(self.seed)
        current_seq = full_seq.upper().replace('U', 'T')
        if not 0 <= cds_start < cds_end <= len(current_seq):
            raise ValueError("CDS coordinates must satisfy 0 <= cds_start < cds_end <= sequence length.")
        if (cds_end - cds_start) % 3 != 0:
            raise ValueError("CDS length must be divisible by 3.")
        if min_gain < 0:
            raise ValueError("min_gain must be non-negative.")
        if not 0.0 <= drop_tolerance <= 1.0:
            raise ValueError("drop_tolerance must be between 0 and 1.")
        if any(weight < 0.0 for weight in (
            penalty_weight, consistency_weight, ratio_weight,
            utr5_penalty_weight, aug_context_weight, specificity_weight,
        )):
            raise ValueError("Fitness weights must be non-negative.")
        if batch_size < 1 or beam_width < 1:
            raise ValueError("batch_size and beam_width must be positive.")
        if iterations < 0 or patience < 0:
            raise ValueError("iterations and patience must be non-negative.")
        if max_candidate_attempts is not None and max_candidate_attempts < 1:
            raise ValueError("max_candidate_attempts must be positive when provided.")
        if max_lineage_depth < 1:
            raise ValueError("max_lineage_depth must be at least 1.")
        mutation_fraction_limits = (
            max_utr5_mutation_fraction,
            max_cds_mutation_fraction,
            max_utr3_mutation_fraction,
            max_total_mutation_fraction,
        )
        if any(not 0.0 <= value <= 1.0 for value in mutation_fraction_limits):
            raise ValueError("Cumulative mutation fractions must be between 0 and 1.")
        if not 0.0 <= high_te_exclusion_fraction < 1.0:
            raise ValueError("high_te_exclusion_fraction must be in [0, 1).")

        if mutation_rate is not None:
            utr5_base_rate = mutation_rate
            utr3_base_rate = mutation_rate
        if cds_mutation_rate is not None:
            cds_base_rate = cds_mutation_rate

        utr5_length = cds_start
        utr3_length = len(current_seq) - cds_end
        cds_codon_count = (cds_end - cds_start) // 3
        cds_mutable_codon_count = 0
        for codon_start in range(cds_start, cds_end, 3):
            codon = current_seq[codon_start:codon_start + 3]
            aa = self.STANDARD_GENETIC_CODE.get(codon)
            if aa is not None and len(self.aa_to_codons[aa]) > 1:
                cds_mutable_codon_count += 1
        utr5_effective_rate, utr5_expected_burden = self._bounded_mutation_rate(
            utr5_length, utr5_base_rate,
            utr5_min_expected_mutations, utr5_max_expected_mutations,
        )
        utr3_effective_rate, utr3_expected_burden = self._bounded_mutation_rate(
            utr3_length, utr3_base_rate,
            utr3_min_expected_mutations, utr3_max_expected_mutations,
        )
        cds_effective_rate, cds_expected_burden = self._bounded_mutation_rate(
            cds_mutable_codon_count, cds_base_rate,
            cds_min_expected_mutations, cds_max_expected_mutations,
        )
        operator_weights = self._normalized_operator_weights(
            mode,
            cds_fraction=global_cds_fraction,
            utr5_fraction=global_utr5_fraction,
            utr3_fraction=global_utr3_fraction,
            joint_fraction=global_joint_fraction,
            utr_only_utr5_fraction=utr_only_utr5_fraction,
            utr_only_utr3_fraction=utr_only_utr3_fraction,
            utr_only_joint_fraction=utr_only_joint_fraction,
        )
        if mode == 2 and cds_mutable_codon_count == 0:
            raise ValueError("CDS-only mode requires at least one synonymous-mutable codon.")
        if mode == 3 and utr5_length == 0:
            raise ValueError("5'UTR-only mode requires a non-empty 5'UTR.")
        if mode == 4 and utr3_length == 0:
            raise ValueError("3'UTR-only mode requires a non-empty 3'UTR.")
        if mode == 5 and utr5_length + utr3_length == 0:
            raise ValueError("UTR-only mode requires at least one non-empty UTR.")

        active_units_by_mode = {
            1: utr5_length + cds_mutable_codon_count + utr3_length,
            2: cds_mutable_codon_count,
            3: utr5_length,
            4: utr3_length,
            5: utr5_length + utr3_length,
        }
        active_unit_count = active_units_by_mode[mode]

        def mutation_limit(unit_count: int, fraction: float) -> int:
            if unit_count <= 0 or fraction <= 0.0:
                return 0
            return min(unit_count, int(math.ceil(unit_count * fraction)))

        mutation_limits = {
            "utr5_edits": mutation_limit(utr5_length, max_utr5_mutation_fraction),
            "cds_codon_edits": mutation_limit(
                cds_mutable_codon_count, max_cds_mutation_fraction
            ),
            "utr3_edits": mutation_limit(utr3_length, max_utr3_mutation_fraction),
            "total_design_edits": mutation_limit(
                active_unit_count, max_total_mutation_fraction
            ),
        }

        def mutation_load(seq: str) -> Dict[str, Any]:
            return self._mutation_load(
                seq,
                current_seq,
                cds_start,
                cds_end,
                cds_mutable_codon_count,
                active_unit_count,
            )

        def within_mutation_budget(seq: str) -> bool:
            load = mutation_load(seq)
            return all(
                load[name] <= limit for name, limit in mutation_limits.items()
            )

        available_region_count = sum((
            cds_mutable_codon_count > 0,
            utr5_length > 0,
            utr3_length > 0,
        ))
        available_utr_count = sum((utr5_length > 0, utr3_length > 0))
        operator_available = {
            "cds": cds_mutable_codon_count > 0,
            "utr5": utr5_length > 0,
            "utr3": utr3_length > 0,
            "joint": available_region_count >= 2,
            "utr_joint": available_utr_count >= 2,
        }
        operator_weights = {
            name: weight for name, weight in operator_weights.items()
            if operator_available[name]
        }
        operator_weight_sum = sum(operator_weights.values())
        if operator_weight_sum <= 0.0:
            raise ValueError("No mutation operator is available for the supplied sequence regions.")
        operator_weights = {
            name: weight / operator_weight_sum
            for name, weight in operator_weights.items()
        }
        
        mode_str = {1: "Global", 2: "CDS", 3: "5'UTR", 4: "3'UTR", 5: "UTRs Only"}.get(mode, str(mode))
        print(f"Initializing Specificity Optimization | Mode: {mode_str} | Batch Size: {batch_size}")
        print(
            "Length-adaptive mutation settings | "
            f"5'UTR: L={utr5_length}, rate={utr5_effective_rate:.6f}, E[attempts]={utr5_expected_burden:.3f} | "
            f"3'UTR: L={utr3_length}, rate={utr3_effective_rate:.6f}, E[attempts]={utr3_expected_burden:.3f} | "
            f"CDS: codons={cds_codon_count}, mutable={cds_mutable_codon_count}, "
            f"rate={cds_effective_rate:.6f}, E[attempts]={cds_expected_burden:.3f}"
        )
        print(
            "Mutation operator mixture: "
            + ", ".join(f"{name}={weight:.1%}" for name, weight in operator_weights.items())
        )
        print(
            "Cumulative WT-relative mutation limits | "
            f"5'UTR nt={mutation_limits['utr5_edits']} | "
            f"CDS codons={mutation_limits['cds_codon_edits']} | "
            f"3'UTR nt={mutation_limits['utr3_edits']} | "
            f"total active-unit edits={mutation_limits['total_design_edits']}/{active_unit_count} | "
            f"maximum lineage depth={max_lineage_depth}"
        )
        if self.valid_negative_cell_types:
            print(f"Target: {self.target_cell_type} | Negative Targets: {', '.join(self.valid_negative_cell_types)} | Spec. Weight: {specificity_weight}")

        wt_target_signal = self.predict_signal(current_seq, self.target_cell_type)
        valid_end = min(len(wt_target_signal), cds_end)
        wt_profile_full = wt_target_signal[cds_start:valid_end:3]
        if wt_profile_full.size == 0:
            raise ValueError("The model output does not contain a valid CDS frame-0 profile.")

        eps = 1e-6
        self.wt_target_te = float(np.mean(wt_profile_full))
        if self.wt_target_te <= eps:
            raise ValueError("WT target TE is zero in linear space; fold-change optimization is undefined.")
        self.wt_cv = float(np.std(wt_profile_full) / (self.wt_target_te + eps))
        self.wt_utr5_mean = float(np.mean(wt_target_signal[:cds_start])) if cds_start > 0 else 0.0
        self.wt_cds_utr_ratio = (
            self.wt_target_te / max(self.wt_utr5_mean, eps) if cds_start > 0 else 0.0
        )
        wt_aug_end = min(len(wt_target_signal), cds_start + 30)
        wt_aug_mean = (
            float(np.mean(wt_target_signal[cds_start:wt_aug_end:3]))
            if wt_aug_end > cds_start else self.wt_target_te
        )
        self.wt_aug_context_ratio = wt_aug_mean / (self.wt_target_te + eps)
        wt_profile = wt_profile_full if use_continuity_constraint else None

        wt_neg_list = [
            self.predict_te(current_seq, cds_start, cds_end, cell_type=cell_type)
            for cell_type in self.valid_negative_cell_types
        ]
        self.wt_max_neg_te = np.max(wt_neg_list) if wt_neg_list else 1.0
        print(f"WT Target={self.wt_target_te:.4f} WT Worst-case Neg={self.wt_max_neg_te:.4f}")
        min_acceptable_te = self.wt_target_te * (1.0 + min_gain)
        print(f"Minimum acceptable target TE={min_acceptable_te:.4f} ({min_gain:.2%} above WT)")
        
        evaluated_cache = {}

        def get_evaluated_metrics(seq_list: List[str]):
            novel_seqs = [s for s in seq_list if s not in evaluated_cache]
            
            if novel_seqs:
                fits, tes, neg_tes, spec_fcs = self.evaluate_batch(
                    novel_seqs, cds_start, cds_end,
                    wt_profile=wt_profile,
                    drop_tolerance=drop_tolerance,
                    penalty_weight=penalty_weight,
                    consistency_weight=consistency_weight,
                    ratio_weight=ratio_weight,
                    utr5_penalty_weight=utr5_penalty_weight,
                    aug_context_weight=aug_context_weight,
                    specificity_weight=specificity_weight,
                    consistency_tolerance=consistency_tolerance,
                    utr5_tolerance=utr5_tolerance,
                    aug_context_tolerance=aug_context_tolerance,
                )
                for s, f, t, n, sp in zip(novel_seqs, fits, tes, neg_tes, spec_fcs):
                    evaluated_cache[s] = (float(f), float(t), float(n), float(sp))
            
            return [evaluated_cache[s] for s in seq_list]

        wt_metrics = get_evaluated_metrics([current_seq])[0]
        wt_true_fit, wt_true_te = wt_metrics[0], wt_metrics[1]
        print(f"Baseline (WT) Target TE: {wt_true_te:.4f} | Baseline Fitness: {wt_true_fit:.4f}")

        sequence_metadata = {
            current_seq: {
                "parent_sequence": None,
                "generation": 0,
                "lineage_depth": 0,
                "mutation_operator": "wt",
            }
        }
        candidate_archive = {}

        def archive_candidate(seq: str, metric: Tuple[float, float, float, float]) -> None:
            if seq in candidate_archive:
                return
            fit, te, neg_te, spec_fc = metric
            metadata = sequence_metadata[seq]
            parent_sequence = metadata["parent_sequence"]
            load = mutation_load(seq)
            log2_te_gain = float(np.log2(max(te / (wt_true_te + eps), eps)))
            prior_adjusted_gain = float(fit - wt_true_fit)
            candidate_archive[seq] = {
                "candidate_id": len(candidate_archive),
                "parent_candidate_id": (
                    candidate_archive[parent_sequence]["candidate_id"]
                    if parent_sequence is not None and parent_sequence in candidate_archive
                    else None
                ),
                "sequence": seq,
                "generation": metadata["generation"],
                "lineage_depth": metadata["lineage_depth"],
                "mutation_operator": metadata["mutation_operator"],
                "fitness": float(fit),
                "prior_adjusted_gain": prior_adjusted_gain,
                "fitness_adjustment_vs_te": prior_adjusted_gain - log2_te_gain,
                "target_te": float(te),
                "target_te_gain": float(te - wt_true_te),
                "target_te_fold_change": float(te / (wt_true_te + eps)),
                "log2_target_te_gain": log2_te_gain,
                "neg_te": float(neg_te),
                "specificity_fc": float(spec_fc),
                "meets_min_gain": bool(te >= min_acceptable_te),
                "min_acceptable_te": float(min_acceptable_te),
                "within_mutation_budget": True,
                "active_unit_count": active_unit_count,
                "max_utr5_edits": mutation_limits["utr5_edits"],
                "max_cds_codon_edits": mutation_limits["cds_codon_edits"],
                "max_utr3_edits": mutation_limits["utr3_edits"],
                "max_total_design_edits": mutation_limits["total_design_edits"],
                "configured_max_utr5_mutation_fraction": max_utr5_mutation_fraction,
                "configured_max_cds_mutation_fraction": max_cds_mutation_fraction,
                "configured_max_utr3_mutation_fraction": max_utr3_mutation_fraction,
                "configured_max_total_mutation_fraction": max_total_mutation_fraction,
                "configured_max_lineage_depth": max_lineage_depth,
                "configured_high_te_exclusion_fraction": high_te_exclusion_fraction,
                "parameter1_te_rank": None,
                "parameter1_te_percentile": None,
                "high_te_tail_excluded": False,
                "pareto_eligible": False,
                "on_pareto_front": False,
                "selected_as_pareto_knee": False,
                **load,
            }

        archive_candidate(current_seq, wt_metrics)

        pool = [(current_seq, wt_true_fit, wt_true_te)]
        
        best_te_hist, curr_te_hist = [], []
        search_best_fit = wt_true_fit
        global_best_fit = None
        global_best_te = None
        patience_counter = 0

        for i in range(iterations):
            candidates_set = {p[0] for p in pool}

            eligible_pool = [
                record for record in pool
                if sequence_metadata[record[0]]["lineage_depth"] < max_lineage_depth
            ]
            if not eligible_pool:
                print(
                    f"Lineage-depth stopping triggered before iteration {i:03d}: "
                    f"all beam parents reached depth {max_lineage_depth}."
                )
                break

            # ==========================================
            # Roulette Wheel Sampling for balanced mutations
            # ==========================================
            pool_fits = np.array([p[1] for p in eligible_pool])
            if np.max(pool_fits) == np.min(pool_fits):
                parent_probs = np.ones(len(pool_fits)) / len(pool_fits)
            else:
                shifted_fits = pool_fits - np.min(pool_fits) + 1e-6
                parent_probs = shifted_fits / shifted_fits.sum()
            
            attempt_limit = max_candidate_attempts or max(batch_size * 50, 1000)
            attempts = 0
            requested_new_candidates = max(0, batch_size - len(candidates_set))
            operator_counts = self._allocate_operator_counts(
                requested_new_candidates, operator_weights
            )
            successful_operator_counts = {name: 0 for name in operator_counts}
            if i == 0:
                print(
                    "Planned new-candidate quota per batch: "
                    + ", ".join(f"{name}={count}" for name, count in operator_counts.items())
                )
            operator_schedule = [
                operator
                for operator, count in operator_counts.items()
                for _ in range(count)
            ]
            random.shuffle(operator_schedule)
            per_slot_attempt_limit = max(
                10,
                min(100, attempt_limit // max(1, requested_new_candidates)),
            )
            budget_rejections = 0

            for operator in operator_schedule:
                for _ in range(per_slot_attempt_limit):
                    if attempts >= attempt_limit:
                        break
                    attempts += 1
                    parent_idx = np.random.choice(len(eligible_pool), p=parent_probs)
                    parent_seq = eligible_pool[parent_idx][0]
                    mutant = self._mutate_sequence(
                        parent_seq, cds_start, cds_end, operator=operator,
                        utr5_mutation_rate=utr5_effective_rate,
                        utr3_mutation_rate=utr3_effective_rate,
                        cds_mutation_rate=cds_effective_rate,
                    )
                    if not within_mutation_budget(mutant):
                        budget_rejections += 1
                        continue
                    if mutant not in candidates_set:
                        candidates_set.add(mutant)
                        sequence_metadata.setdefault(mutant, {
                            "parent_sequence": parent_seq,
                            "generation": i + 1,
                            "lineage_depth": sequence_metadata[parent_seq]["lineage_depth"] + 1,
                            "mutation_operator": operator,
                        })
                        successful_operator_counts[operator] += 1
                        break

            operator_names = list(operator_weights)
            while len(candidates_set) < batch_size and attempts < attempt_limit:
                attempts += 1
                operator_deficits = np.array([
                    max(0, operator_counts[name] - successful_operator_counts[name])
                    for name in operator_names
                ], dtype=float)
                if operator_deficits.sum() > 0:
                    operator_probabilities = operator_deficits / operator_deficits.sum()
                else:
                    operator_probabilities = np.array([
                        operator_weights[name] for name in operator_names
                    ])
                operator = str(np.random.choice(operator_names, p=operator_probabilities))
                parent_idx = np.random.choice(len(eligible_pool), p=parent_probs)
                parent_seq = eligible_pool[parent_idx][0]
                mutant = self._mutate_sequence(
                    parent_seq, cds_start, cds_end, operator=operator,
                    utr5_mutation_rate=utr5_effective_rate,
                    utr3_mutation_rate=utr3_effective_rate,
                    cds_mutation_rate=cds_effective_rate,
                )
                if not within_mutation_budget(mutant):
                    budget_rejections += 1
                    continue
                if mutant not in candidates_set:
                    candidates_set.add(mutant)
                    sequence_metadata.setdefault(mutant, {
                        "parent_sequence": parent_seq,
                        "generation": i + 1,
                        "lineage_depth": sequence_metadata[parent_seq]["lineage_depth"] + 1,
                        "mutation_operator": operator,
                    })
                    successful_operator_counts[operator] += 1
            if len(candidates_set) < batch_size:
                print(
                    f"Warning: generated {len(candidates_set)}/{batch_size} unique candidates "
                    f"after {attempt_limit} attempts; mutation-budget rejections={budget_rejections}."
                )

            candidates_list = sorted(candidates_set)
            
            metrics = get_evaluated_metrics(candidates_list)
            
            batch_records = []
            feasible_improved = False
            for seq, metric in zip(candidates_list, metrics):
                fit, te, neg_te, spec_fc = metric
                archive_candidate(seq, metric)
                origin_operator = sequence_metadata[seq]["mutation_operator"]
                batch_records.append((seq, fit, te, origin_operator))

                is_feasible = te >= min_acceptable_te
                if is_feasible and (
                    global_best_fit is None
                    or fit > global_best_fit
                    or (fit == global_best_fit and te > global_best_te)
                ):
                    global_best_fit = fit
                    global_best_te = te
                    feasible_improved = True
                    print(
                        f"Iteration {i:03d} [{mode_str} Focus]: Feasible Variant Identified | "
                        f"Fitness: {global_best_fit:.4f} | Target TE: {global_best_te:.4f}"
                    )

            batch_records.sort(key=lambda x: (x[1], x[2]), reverse=True)
            
            search_improved = batch_records[0][1] > search_best_fit
            if search_improved:
                search_best_fit = batch_records[0][1]

            if search_improved or feasible_improved:
                patience_counter = 0
            else:
                patience_counter += 1

            if patience > 0 and patience_counter >= patience:
                print(f"Early stopping triggered at iteration {i:03d}. No improvement for {patience} consecutive iterations.")
                break
                
            # ==========================================
            # Soft Beam Search to avoid local optima
            # ==========================================
            next_pool = [batch_records[0]]
            selected_pool_sequences = {batch_records[0][0]}

            if beam_width > 1 and len(operator_weights) > 1:
                operators_by_priority = sorted(
                    operator_weights,
                    key=lambda name: operator_weights[name],
                    reverse=True,
                )
                for operator in operators_by_priority:
                    if len(next_pool) >= beam_width:
                        break
                    operator_candidate = next((
                        record for record in batch_records
                        if record[3] == operator and record[0] not in selected_pool_sequences
                    ), None)
                    if operator_candidate is not None:
                        next_pool.append(operator_candidate)
                        selected_pool_sequences.add(operator_candidate[0])

            remaining_slots = beam_width - len(next_pool)
            if remaining_slots > 0 and len(batch_records) > len(next_pool):
                pool_candidates = [
                    record for record in batch_records[:beam_width * 5]
                    if record[0] not in selected_pool_sequences
                ]
                fits = np.array([r[1] for r in pool_candidates])
                if len(pool_candidates) > 0:
                    temperature = 0.2
                    scaled_fits = fits / temperature
                    scaled_fits = scaled_fits - np.max(scaled_fits)
                    probs = np.exp(scaled_fits) / np.sum(np.exp(scaled_fits))

                    sample_size = min(remaining_slots, len(pool_candidates))
                    sampled_indices = np.random.choice(
                        len(pool_candidates), size=sample_size, p=probs, replace=False
                    )

                    for idx in sampled_indices:
                        next_pool.append(pool_candidates[idx])
            
            pool = next_pool
            best_te_hist.append(global_best_te if global_best_te is not None else wt_true_te)
            curr_te_hist.append(pool[0][2])

        feasible_records = [
            record for record in candidate_archive.values()
            if record["meets_min_gain"] and record["mutation_operator"] != "wt"
        ]
        if not feasible_records:
            print(
                "Optimization completed without a candidate meeting the minimum TE gain. "
                "Returning WT as a safe fallback while preserving the complete candidate archive."
            )
            trajectory_history = sorted(
                candidate_archive.values(),
                key=lambda record: (
                    record["generation"],
                    record["total_design_edits"],
                    -record["target_te"],
                    record["sequence"],
                ),
            )
            return current_seq, wt_true_te, trajectory_history, best_te_hist, curr_te_hist

        print(
            f"Search completed with {len(candidate_archive)} archived candidates and "
            f"{len(feasible_records)} candidates meeting the minimum TE gain."
        )

        ranked_by_te = sorted(
            feasible_records,
            key=lambda record: (-record["target_te"], -record["fitness"], record["sequence"]),
        )
        for rank, record in enumerate(ranked_by_te, start=1):
            record["parameter1_te_rank"] = rank
            record["parameter1_te_percentile"] = (
                100.0 * (len(ranked_by_te) - rank + 1) / len(ranked_by_te)
            )
        excluded_count = 0
        if len(ranked_by_te) > 1 and high_te_exclusion_fraction > 0.0:
            excluded_count = min(
                len(ranked_by_te) - 1,
                int(math.ceil(len(ranked_by_te) * high_te_exclusion_fraction)),
            )
        for record in ranked_by_te[:excluded_count]:
            record["high_te_tail_excluded"] = True

        pareto_candidates = ranked_by_te[excluded_count:]
        for record in pareto_candidates:
            record["pareto_eligible"] = True

        front_indices = self._pareto_front_indices(pareto_candidates)
        for idx in front_indices:
            pareto_candidates[idx]["on_pareto_front"] = True
        knee_idx = self._select_pareto_knee(pareto_candidates, front_indices)
        selected_record = pareto_candidates[knee_idx]
        selected_record["selected_as_pareto_knee"] = True

        print(
            f"Safe selection excluded the highest-TE {excluded_count}/{len(ranked_by_te)} "
            f"feasible candidates ({high_te_exclusion_fraction:.1%} target), retained "
            f"{len(front_indices)} Pareto-front candidates, and selected candidate "
            f"{selected_record['candidate_id']} at lineage depth "
            f"{selected_record['lineage_depth']} with TE={selected_record['target_te']:.4f}, "
            f"fitness gain={selected_record['prior_adjusted_gain']:.4f}, and mutation "
            f"burden={selected_record['total_mutation_burden']:.4f}."
        )

        trajectory_history = sorted(
            candidate_archive.values(),
            key=lambda record: (
                record["generation"],
                record["total_design_edits"],
                -record["target_te"],
                record["sequence"],
            ),
        )
        return (
            selected_record["sequence"],
            selected_record["target_te"],
            trajectory_history,
            best_te_hist,
            curr_te_hist,
        )

# Visualization and Data Export
def plot_convergence(best_hist: list, curr_hist: list, wt_te: float, out_path: str):
    plt.figure(figsize=(10, 6))
    plt.plot(curr_hist, label='Current Generation Best Target TE', color='#FFA500', alpha=0.5, linewidth=1)
    plt.plot(best_hist, label='Best Feasible-Fitness Candidate Target TE', color='#FF0000', linewidth=2.5)
    plt.axhline(y=wt_te, color='#0000FF', linestyle='--', linewidth=2, label=f'WT Baseline ({wt_te:.4f})')
    
    plt.title('Sequence Optimization Convergence Trajectory', fontsize=14, fontweight='bold')
    plt.xlabel('Iteration / Epoch', fontsize=12)
    plt.ylabel('Target Cell Translation Efficiency (TE)', fontsize=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Convergence plot successfully exported to: {out_path}")


# Command Line Interface Execution
def main():
    parser = argparse.ArgumentParser(description="mRNA Sequence TE Specificity Optimizer (Evolutionary Framework)")
    default_model_config = os.path.join(
        SRC_DIR, "config/base_model_384d_16h_12l_64env_16ad_bs.yaml"
    )
    
    default_5utr = "AGAATAAACTAGTATTCTTCTGGTCCCCACAGACTCAGAGAGAACCCggatccgccacc".upper()
    default_cds = "AUGGGAGUCAAAGUUCUGUUUGCCCUGAUCUGCAUCGCUGUGGCCGAGGCCAAGCCCACCGAGAACAACGAAGACUUCAACAUCGUGGCCGUGGCCAGCAACUUCGCGACCACGGAUCUCGAUGCUGACCGCGGGAAGUUGCCCGGCAAGAAGCUGCCGCUGGAGGUGCUCAAAGAGAUGGAAGCCAAUGCCCGGAAAGCUGGCUGCACCAGGGGCUGUCUGAUCUGCCUGUCCCACAUCAAGUGCACGCCCAAGAUGAAGAAGUUCAUCCCAGGACGCUGCCACACCUACGAAGGCGACAAAGAGUCCGCACAGGGCGGCAUAGGCGAGGCGAUCGUCGACAUUCCUGAGAUUCCUGGGUUCAAGGACUUGGAGCCCAUGGAGCAGUUCAUCGCACAGGUCGAUCUGUGUGUGGACUGCACAACUGGCUGCCUCAAAGGGCUUGCCAACGUGCAGUGUUCUGACCUGCUCAAGAAGUGGCUGCCGCAACGCUGUGCGACCUUUGCCAGCAAGAUCCAGGGCCAGGUGGACAAGAUCAAGGGGGCCGGUGGUGACUAA".upper()
    default_cds = default_cds.replace('U', 'T')
    default_3utr = "taaCTCGAGCTGGTACTGCATGCACGCAATGCTAGCTGCCCCTTTCCCGTCCTGGGTACCCCGAGTCTCCCCCGACCTCGGGTCCCAGGTATGCTCCCACCTCCACCTGCCCCACTCACCACCTCTGCTAGTTCCAGACACCTCCCAAGCACGCAGCAATGCAGCTCAAAACGCTTAGCCTAGCCACACCCCCACGGGAAACAGCAGTGATTAACCTTTAGCAATAAACGAAAGTTTAACTAAGCTATACTAACCCCAGGGTTGGTCAATTTCGTGCCAGCCACACCCTGGAGCTAGC".upper()

    # --- New Arguments for Model, Species, and Environment Configs ---
    parser.add_argument("--model_config", type=str,
                        default=default_model_config,
                        help="Path to the sequence-only BaseModel configuration YAML file")
    parser.add_argument("--model_weights", type=str, required=True,
                        help="Path to the pretrained model weights (.pt file); no default checkpoint is used")
    parser.add_argument(
        "--head_type", type=str, required=True,
        choices=["translation_profile", "psite_density"],
        help="Prediction head architecture used to train the supplied checkpoint"
    )
    parser.add_argument("--species", type=str, default="human", help="Species string passed to the model (e.g., 'human')")
    parser.add_argument("--expr_dict", type=str, 
                        default="/home/user/data3/yaoc/translation_model/data/lib/human_expression_dict.pt", 
                        help="Path to the cell-type expression dictionary (.pt file)")

    parser.add_argument("--utr5", type=str, default=default_5utr, help="5' UTR Sequence")
    parser.add_argument("--cds", type=str, default=default_cds, help="CDS Sequence")
    parser.add_argument("--utr3", type=str, default=default_3utr, help="3' UTR Sequence")
    parser.add_argument("--mode", type=int, default=1, choices=[1, 2, 3, 4, 5], help="Optimization Mode")
    
    # Target and Specificity Arguments
    parser.add_argument("--target_cell", type=str, default="HEK293T_inhouse", help="Target cell type for high expression")
    parser.add_argument("--neg_cells", type=str, nargs='*', default=[], help="List of non-target cell types for low expression (space separated)")
    parser.add_argument("--spec_weight", type=float, default=1.0, help="Penalty weight applied to expression in negative cell types")

    parser.add_argument("--iter", type=int, default=800, help="Number of maximum iterations")
    parser.add_argument("--batch", type=int, default=128, help="Batch size per iteration")
    parser.add_argument("--beam_width", type=int, default=8, help="Number of seed sequences retained per generation")
    parser.add_argument("--utr5_base_rate", type=float, default=0.008, help="Base per-nucleotide 5'UTR mutation rate before burden bounding")
    parser.add_argument("--utr3_base_rate", type=float, default=0.002, help="Base per-nucleotide 3'UTR mutation rate before burden bounding")
    parser.add_argument("--cds_base_rate", type=float, default=0.01, help="Base per-codon CDS mutation rate before burden bounding")
    parser.add_argument("--utr5_min_edits", type=float, default=0.5, help="Minimum expected 5'UTR mutation burden per operator call")
    parser.add_argument("--utr5_max_edits", type=float, default=2.0, help="Maximum expected 5'UTR mutation burden per operator call")
    parser.add_argument("--utr3_min_edits", type=float, default=0.5, help="Minimum expected 3'UTR mutation burden per operator call")
    parser.add_argument("--utr3_max_edits", type=float, default=2.0, help="Maximum expected 3'UTR mutation burden per operator call")
    parser.add_argument("--cds_min_edits", type=float, default=1.0, help="Minimum expected CDS codon mutation burden per operator call")
    parser.add_argument("--cds_max_edits", type=float, default=3.0, help="Maximum expected CDS codon mutation burden per operator call")
    parser.add_argument("--global_cds_fraction", type=float, default=0.40, help="Global-mode fraction assigned to the CDS-only operator")
    parser.add_argument("--global_utr5_fraction", type=float, default=0.25, help="Global-mode fraction assigned to the 5'UTR-only operator")
    parser.add_argument("--global_utr3_fraction", type=float, default=0.15, help="Global-mode fraction assigned to the 3'UTR-only operator")
    parser.add_argument("--global_joint_fraction", type=float, default=0.20, help="Global-mode fraction assigned to the joint operator")
    parser.add_argument("--utr_only_utr5_fraction", type=float, default=0.60, help="UTR-only-mode fraction assigned to the 5'UTR-only operator")
    parser.add_argument("--utr_only_utr3_fraction", type=float, default=0.25, help="UTR-only-mode fraction assigned to the 3'UTR-only operator")
    parser.add_argument("--utr_only_joint_fraction", type=float, default=0.15, help="UTR-only-mode fraction assigned to the UTR-joint operator")
    parser.add_argument("--mut_rate", type=float, default=None, help="Deprecated compatibility option overriding both UTR base rates")
    parser.add_argument("--cds_mut_rate", type=float, default=None, help="Deprecated compatibility option overriding the CDS base rate")
    parser.add_argument("--min_gain", type=float, default=0.01, help="Minimum required target TE gain over WT")
    parser.add_argument("--patience", type=int, default=50, help="Number of consecutive epochs without improvement to stop training")
    parser.add_argument("--max_lineage_depth", type=int, default=10, help="Maximum number of mutation steps from WT in any candidate lineage")
    parser.add_argument("--max_utr5_mutation_fraction", type=float, default=0.10, help="Maximum cumulative 5'UTR nucleotide edit fraction relative to WT")
    parser.add_argument("--max_cds_mutation_fraction", type=float, default=0.10, help="Maximum cumulative CDS codon edit fraction relative to mutable WT codons")
    parser.add_argument("--max_utr3_mutation_fraction", type=float, default=0.10, help="Maximum cumulative 3'UTR nucleotide edit fraction relative to WT")
    parser.add_argument("--max_total_mutation_fraction", type=float, default=0.10, help="Maximum cumulative edit fraction across active design units")
    parser.add_argument("--high_te_exclusion_fraction", type=float, default=0.20, help="Fraction of highest-TE feasible candidates excluded before Pareto selection")
    parser.add_argument("--continuity_weight", type=float, default=0.2, help="Weight for relative WT-profile drop penalty")
    parser.add_argument("--drop_tolerance", type=float, default=0.5, help="Allowed per-position drop relative to the WT profile")
    parser.add_argument("--consistency_weight", type=float, default=0.25, help="Weight for relative CDS CV increase penalty")
    parser.add_argument("--ratio_weight", type=float, default=0.05, help="Weight for relative CDS/UTR ratio improvement bonus")
    parser.add_argument("--utr5_penalty_weight", type=float, default=0.1, help="Weight for relative 5'UTR signal increase penalty")
    parser.add_argument("--aug_context_weight", type=float, default=0.1, help="Weight for relative AUG-context increase penalty")
    parser.add_argument("--consistency_tolerance", type=float, default=0.1, help="Allowed relative CDS CV increase over WT")
    parser.add_argument("--utr5_tolerance", type=float, default=0.1, help="Allowed relative 5'UTR signal increase over WT")
    parser.add_argument("--aug_context_tolerance", type=float, default=0.1, help="Allowed relative AUG-context ratio increase over WT")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed for reproducibility")
    parser.add_argument("--outdir", type=str, default="/home/user/data3/yaoc/translation_model/code/TE_optimization/res", help="Output directory path")
    parser.add_argument("--prefix", type=str, default="optimized_Gluc", help="Prefix for output files")
    
    args = parser.parse_args()

    utr5_sequence = args.utr5.upper().replace('U', 'T')
    cds_sequence = args.cds.upper().replace('U', 'T')
    utr3_sequence = args.utr3.upper().replace('U', 'T')
    full_sequence = utr5_sequence + cds_sequence + utr3_sequence
    cds_start_idx = len(utr5_sequence)
    cds_end_idx = cds_start_idx + len(cds_sequence)

    print("Sequence Specificity Optimization Pipeline Initialized")
    print(f"Global Random Seed: {args.seed}")

    # ==========================================================
    # Initialize Model dynamically using parsed CLI arguments
    # ==========================================================
    print(f"Loading Base Model Configuration from: {args.model_config}")
    base_model = BaseModel.from_config(args.model_config).to('cuda')
    
    print(f"Initializing prediction head: {args.head_type}")
    if args.head_type == "translation_profile":
        prediction_head = TranslationProfileHead.create_from_model(base_model, d_pred_h=384)
    else:
        prediction_head = PsiteDensityHead.create_from_model(base_model, d_pred_h=384)
    base_model.add_head(
        "count",
        prediction_head,
        overwrite=True
    )

    print(f"Loading Pretrained Model Checkpoint from: {args.model_weights}")
    base_model.load_pretrained_weights(args.model_weights, strict=True)
    base_model.eval()  # Use deterministic inference behavior.

    # Initialize the optimizer with dynamically parsed arguments
    optimizer = BatchedBeamOptimizer(
        model=base_model, 
        tokenizer_fn=tokenize_seq_onehot,
        target_cell_type=args.target_cell, 
        negative_cell_types=args.neg_cells,
        species=args.species,                     
        head_name="count",
        expr_dict_path=args.expr_dict,            
        seed=args.seed 
    )
    
    wt_target_te = optimizer.predict_te(full_sequence, cds_start_idx, cds_end_idx, cell_type=args.target_cell)
    print(f"Initial Wild-Type (WT) Target TE Score ({args.target_cell}): {wt_target_te:.4f}")
    
    for neg_cell in optimizer.valid_negative_cell_types:
        wt_neg_te = optimizer.predict_te(full_sequence, cds_start_idx, cds_end_idx, cell_type=neg_cell)
        print(f"Initial Wild-Type (WT) Negative TE Score ({neg_cell}): {wt_neg_te:.4f}")

    optimized_sequence, final_te_score, history_list, best_hist, curr_hist = optimizer.optimize(
        full_seq=full_sequence,
        cds_start=cds_start_idx,
        cds_end=cds_end_idx,
        mode=args.mode,             
        iterations=args.iter,     
        mutation_rate=args.mut_rate,
        cds_mutation_rate=args.cds_mut_rate,
        utr5_base_rate=args.utr5_base_rate,
        utr3_base_rate=args.utr3_base_rate,
        cds_base_rate=args.cds_base_rate,
        utr5_min_expected_mutations=args.utr5_min_edits,
        utr5_max_expected_mutations=args.utr5_max_edits,
        utr3_min_expected_mutations=args.utr3_min_edits,
        utr3_max_expected_mutations=args.utr3_max_edits,
        cds_min_expected_mutations=args.cds_min_edits,
        cds_max_expected_mutations=args.cds_max_edits,
        global_cds_fraction=args.global_cds_fraction,
        global_utr5_fraction=args.global_utr5_fraction,
        global_utr3_fraction=args.global_utr3_fraction,
        global_joint_fraction=args.global_joint_fraction,
        utr_only_utr5_fraction=args.utr_only_utr5_fraction,
        utr_only_utr3_fraction=args.utr_only_utr3_fraction,
        utr_only_joint_fraction=args.utr_only_joint_fraction,
        batch_size=args.batch,
        beam_width=args.beam_width,
        drop_tolerance=args.drop_tolerance,
        penalty_weight=args.continuity_weight,
        consistency_weight=args.consistency_weight,
        ratio_weight=args.ratio_weight,
        utr5_penalty_weight=args.utr5_penalty_weight,
        aug_context_weight=args.aug_context_weight,
        specificity_weight=args.spec_weight,
        consistency_tolerance=args.consistency_tolerance,
        utr5_tolerance=args.utr5_tolerance,
        aug_context_tolerance=args.aug_context_tolerance,
        min_gain=args.min_gain,
        patience=args.patience,
        max_lineage_depth=args.max_lineage_depth,
        max_utr5_mutation_fraction=args.max_utr5_mutation_fraction,
        max_cds_mutation_fraction=args.max_cds_mutation_fraction,
        max_utr3_mutation_fraction=args.max_utr3_mutation_fraction,
        max_total_mutation_fraction=args.max_total_mutation_fraction,
        high_te_exclusion_fraction=args.high_te_exclusion_fraction,
    )

    total_seqs = len(history_list)
    print(f"Optimization concluded with {total_seqs} unique archived candidates.")

    pareto_records = sorted(
        [record for record in history_list if record["on_pareto_front"]],
        key=lambda record: (
            record["total_mutation_burden"],
            -record["prior_adjusted_gain"],
        ),
    )
    selected_records = [
        record for record in history_list if record["selected_as_pareto_knee"]
    ]

    # Data Export
    os.makedirs(args.outdir, exist_ok=True)

    output_fasta = os.path.join(args.outdir, f"{args.prefix}_M{args.mode}_Spec_seed{args.seed}.fasta")
    output_csv = os.path.join(args.outdir, f"{args.prefix}_M{args.mode}_Spec_seed{args.seed}.csv")
    archive_csv = os.path.join(
        args.outdir,
        f"{args.prefix}_M{args.mode}_Spec_seed{args.seed}_all_candidates.csv",
    )
    output_png = os.path.join(args.outdir, f"{args.prefix}_M{args.mode}_Spec_seed{args.seed}_convergence.png")

    pd.DataFrame(history_list).to_csv(archive_csv, index=False)
    results_list = []

    with open(output_fasta, "w") as f_out:
        for data in pareto_records:
            label = (
                "Selected_Pareto_Knee"
                if data["selected_as_pareto_knee"]
                else f"Pareto_Candidate_{data['candidate_id']}"
            )
            f_out.write(
                f">{label}"
                f" | Candidate_ID={data['candidate_id']}"
                f" | Generation={data['generation']}"
                f" | Lineage_Depth={data['lineage_depth']}"
                f" | Fitness={data['fitness']:.4f}"
                f" | Prior_Adjusted_Gain={data['prior_adjusted_gain']:.4f}"
                f" | Target_TE={data['target_te']:.4f}"
                f" | Target_TE_FC={data['target_te_fold_change']:.4f}"
                f" | Mutation_Burden={data['total_mutation_burden']:.4f}"
                f" | Total_Design_Edits={data['total_design_edits']}"
                f" | Operator={data['mutation_operator']}"
                f" | Max_Neg_TE={data['neg_te']:.4f}"
                f" | SpecFC={data['specificity_fc']:.4f}\n"
            )

            f_out.write(f"{data['sequence']}\n")

            row_dict = {
                "Variant_Name": label,
                **data,
            }

            for neg_cell in args.neg_cells:

                neg_score = optimizer.predict_te(
                    data["sequence"],
                    cds_start_idx,
                    cds_end_idx,
                    cell_type=neg_cell
                )

                row_dict[f"Neg_TE_{neg_cell}"] = neg_score

            results_list.append(row_dict)

    df_results = pd.DataFrame(results_list)
    df_results.to_csv(output_csv, index=False)

    if selected_records:
        selected = selected_records[0]
        print(
            "Selected Pareto knee: "
            f"candidate={selected['candidate_id']} | TE={selected['target_te']:.4f} | "
            f"TE_FC={selected['target_te_fold_change']:.4f} | "
            f"prior-adjusted gain={selected['prior_adjusted_gain']:.4f} | "
            f"mutation burden={selected['total_mutation_burden']:.4f}"
        )
    else:
        print("No feasible candidate remained for Pareto selection; WT was returned.")

    plot_convergence(best_hist, curr_hist, wt_target_te, output_png)

    print(
        f"Pipeline executed successfully. Pareto FASTA/CSV and the complete archive "
        f"were saved to: {args.outdir}"
    )

if __name__ == "__main__":
    main()
