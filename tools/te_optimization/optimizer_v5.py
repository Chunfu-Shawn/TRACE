#!/home/user/data2/rbase/envs/anaconda3/envs/ribo_model/bin/python
import os
import random
import math
import argparse
import numpy as np
import torch
import pandas as pd
from typing import Callable, Tuple, List, Optional, Dict
import sys
import matplotlib.pyplot as plt
import heapq

# Model Loading and Environment Configuration
sys.path.append('/public-supool/home/annie/translation_model/TRACE/src')
from model.prediction_heads import PsiteDensityHead
from model.translation_base_model import TranslationBaseModel
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

    def _mutate_sequence(
        self, seq: str, cds_start: int, cds_end: int, mode: int,
        mutation_rate: float, cds_mutation_rate: Optional[float] = None
    ) -> str:
        """Mutate UTR bases and CDS codons using independent per-unit rates."""
        if mode not in {1, 2, 3, 4, 5}:
            raise ValueError(f"Unsupported optimization mode: {mode}")

        utr_rate = float(mutation_rate)
        cds_rate = float(cds_mutation_rate if cds_mutation_rate is not None else mutation_rate)
        if not 0.0 <= utr_rate <= 1.0 or not 0.0 <= cds_rate <= 1.0:
            raise ValueError("Mutation rates must be between 0 and 1.")

        seq_list = list(seq.upper().replace('U', 'T'))
        mutate_cds = mode in {1, 2}
        mutate_utr5 = mode in {1, 3, 5}
        mutate_utr3 = mode in {1, 4, 5}

        if mutate_cds:
            complete_cds_end = cds_start + ((cds_end - cds_start) // 3) * 3
            for codon_start in range(cds_start, complete_cds_end, 3):
                if random.random() > cds_rate:
                    continue
                old_codon = "".join(seq_list[codon_start:codon_start + 3])
                aa = self.STANDARD_GENETIC_CODE.get(old_codon)
                if aa is None:
                    continue
                synonymous_codons = [c for c in self.aa_to_codons[aa] if c != old_codon]
                if synonymous_codons:
                    seq_list[codon_start:codon_start + 3] = list(random.choice(synonymous_codons))

        utr_positions = []
        if mutate_utr5:
            utr_positions.extend(range(0, max(0, min(cds_start, len(seq_list)))))
        if mutate_utr3:
            utr_positions.extend(range(max(0, cds_end), len(seq_list)))

        for pos in utr_positions:
            if random.random() > utr_rate:
                continue
            old_base = seq_list[pos]
            new_base = random.choice([base for base in self.bases if base != old_base])
            if pos < cds_start:
                temp_seq = seq_list.copy()
                temp_seq[pos] = new_base
                local_start = max(0, pos - 2)
                local_end = min(cds_start, pos + 3)
                if "ATG" in "".join(temp_seq[local_start:local_end]):
                    new_base = old_base
            seq_list[pos] = new_base

        return "".join(seq_list)

    def optimize(
        self, full_seq: str, cds_start: int, cds_end: int, 
        mode: int = 1, iterations: int = 200, batch_size: int = 128, 
        mutation_rate: float = 0.05, cds_mutation_rate: float = 0.01,
        beam_width: int = 8, 
        use_continuity_constraint: bool = True, drop_tolerance: float = 0.5,        
        penalty_weight: float = 0.2, consistency_weight: float = 0.25,    
        ratio_weight: float = 0.05, utr5_penalty_weight: float = 0.1,
        aug_context_weight: float = 0.1, specificity_weight: float = 1.0,
        consistency_tolerance: float = 0.1, utr5_tolerance: float = 0.1,
        aug_context_tolerance: float = 0.1, min_gain: float = 0.01,
        patience: int = 50, top_k_size: int = 100,
        max_candidate_attempts: Optional[int] = None
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
        if batch_size < 1 or beam_width < 1 or top_k_size < 1:
            raise ValueError("batch_size, beam_width, and top_k_size must be positive.")
        
        mode_str = {1: "Global", 2: "CDS", 3: "5'UTR", 4: "3'UTR", 5: "UTRs Only"}.get(mode, str(mode))
        print(f"Initializing Specificity Optimization | Mode: {mode_str} | Batch Size: {batch_size}")
        print(f"UTR mutation rate={mutation_rate:.4f} | CDS codon mutation rate={cds_mutation_rate:.4f}")
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

        top_k_heap = []
        seen_in_top_k = set()

        pool = [(current_seq, wt_true_fit, wt_true_te)]
        
        best_te_hist, curr_te_hist = [], []
        search_best_fit = wt_true_fit
        global_best_fit = None
        global_best_te = None
        best_seq = None
        patience_counter = 0

        for i in range(iterations):
            candidates_set = {p[0] for p in pool} 
            
            # ==========================================
            # Roulette Wheel Sampling for balanced mutations
            # ==========================================
            pool_fits = np.array([p[1] for p in pool])
            if np.max(pool_fits) == np.min(pool_fits):
                parent_probs = np.ones(len(pool_fits)) / len(pool_fits)
            else:
                shifted_fits = pool_fits - np.min(pool_fits) + 1e-6
                parent_probs = shifted_fits / shifted_fits.sum()
            
            attempt_limit = max_candidate_attempts or max(batch_size * 50, 1000)
            attempts = 0
            while len(candidates_set) < batch_size and attempts < attempt_limit:
                attempts += 1
                parent_idx = np.random.choice(len(pool), p=parent_probs)
                parent_seq = pool[parent_idx][0]
                mutant = self._mutate_sequence(
                    parent_seq, cds_start, cds_end, mode,
                    mutation_rate=mutation_rate,
                    cds_mutation_rate=cds_mutation_rate,
                )
                candidates_set.add(mutant)
            if len(candidates_set) < batch_size:
                print(
                    f"Warning: generated {len(candidates_set)}/{batch_size} unique candidates "
                    f"after {attempt_limit} attempts."
                )

            candidates_list = sorted(candidates_set)
            
            metrics = get_evaluated_metrics(candidates_list)
            
            batch_records = []
            feasible_improved = False
            for seq, metric in zip(candidates_list, metrics):
                fit, te, neg_te, spec_fc = metric
                batch_records.append((seq, fit, te))

                is_feasible = te >= min_acceptable_te
                if is_feasible and seq not in seen_in_top_k:
                    if len(top_k_heap) < top_k_size:
                        heapq.heappush(top_k_heap, (fit, seq, te, neg_te, spec_fc))
                        seen_in_top_k.add(seq)
                    elif fit > top_k_heap[0][0]:
                        removed = heapq.heappushpop(top_k_heap, (fit, seq, te, neg_te, spec_fc))
                        seen_in_top_k.remove(removed[1])
                        seen_in_top_k.add(seq)

                if is_feasible and (
                    global_best_fit is None
                    or fit > global_best_fit
                    or (fit == global_best_fit and te > global_best_te)
                ):
                    global_best_fit = fit
                    global_best_te = te
                    best_seq = seq
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
            
            if beam_width > 1 and len(batch_records) > 1:
                pool_candidates = batch_records[1: min(len(batch_records), beam_width * 5)] 
                fits = np.array([r[1] for r in pool_candidates])
                
                temperature = 0.2 
                scaled_fits = fits / temperature
                scaled_fits = scaled_fits - np.max(scaled_fits) 
                probs = np.exp(scaled_fits) / np.sum(np.exp(scaled_fits))
                
                sample_size = min(beam_width - 1, len(pool_candidates))
                sampled_indices = np.random.choice(len(pool_candidates), size=sample_size, p=probs, replace=False)
                
                for idx in sampled_indices:
                    next_pool.append(pool_candidates[idx])
            
            pool = next_pool
            best_te_hist.append(global_best_te if global_best_te is not None else wt_true_te)
            curr_te_hist.append(pool[0][2])

        if best_seq is None:
            print(
                "Optimization completed without a candidate meeting the minimum TE gain. "
                "Returning WT as a safe fallback and exporting no optimized variants."
            )
            return current_seq, wt_true_te, [], best_te_hist, curr_te_hist

        print(
            f"Optimization Completed. Feasible fitness: {global_best_fit:.4f} | "
            f"Target TE: {global_best_te:.4f}"
        )
        
        sorted_top_k = sorted(top_k_heap, key=lambda x: x[0], reverse=True)
        trajectory_history = [
            {
                "sequence": item[1],
                "fitness": item[0],
                "target_te": item[2],
                "neg_te": item[3],
                "specificity_fc": item[4],
                "target_te_fold_change": item[2] / (wt_true_te + eps),
                "meets_min_gain": True,
                "min_acceptable_te": min_acceptable_te,
            }
            for item in sorted_top_k
        ]

        return best_seq, global_best_te, trajectory_history, best_te_hist, curr_te_hist

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
    
    default_5utr = "AGAATAAACTAGTATTCTTCTGGTCCCCACAGACTCAGAGAGAACCCggatccgccacc".upper()
    default_cds = "AUGGGAGUCAAAGUUCUGUUUGCCCUGAUCUGCAUCGCUGUGGCCGAGGCCAAGCCCACCGAGAACAACGAAGACUUCAACAUCGUGGCCGUGGCCAGCAACUUCGCGACCACGGAUCUCGAUGCUGACCGCGGGAAGUUGCCCGGCAAGAAGCUGCCGCUGGAGGUGCUCAAAGAGAUGGAAGCCAAUGCCCGGAAAGCUGGCUGCACCAGGGGCUGUCUGAUCUGCCUGUCCCACAUCAAGUGCACGCCCAAGAUGAAGAAGUUCAUCCCAGGACGCUGCCACACCUACGAAGGCGACAAAGAGUCCGCACAGGGCGGCAUAGGCGAGGCGAUCGUCGACAUUCCUGAGAUUCCUGGGUUCAAGGACUUGGAGCCCAUGGAGCAGUUCAUCGCACAGGUCGAUCUGUGUGUGGACUGCACAACUGGCUGCCUCAAAGGGCUUGCCAACGUGCAGUGUUCUGACCUGCUCAAGAAGUGGCUGCCGCAACGCUGUGCGACCUUUGCCAGCAAGAUCCAGGGCCAGGUGGACAAGAUCAAGGGGGCCGGUGGUGACUAA".upper()
    default_cds = default_cds.replace('U', 'T')
    default_3utr = "taaCTCGAGCTGGTACTGCATGCACGCAATGCTAGCTGCCCCTTTCCCGTCCTGGGTACCCCGAGTCTCCCCCGACCTCGGGTCCCAGGTATGCTCCCACCTCCACCTGCCCCACTCACCACCTCTGCTAGTTCCAGACACCTCCCAAGCACGCAGCAATGCAGCTCAAAACGCTTAGCCTAGCCACACCCCCACGGGAAACAGCAGTGATTAACCTTTAGCAATAAACGAAAGTTTAACTAAGCTATACTAACCCCAGGGTTGGTCAATTTCGTGCCAGCCACACCCTGGAGCTAGC".upper()

    # --- New Arguments for Model, Species, and Environment Configs ---
    parser.add_argument("--model_config", type=str, 
                        default="/home/user/data3/rbase/translation_model/models/src/config/base_model_expr_384d_16h_12l_64env_16ad.yaml", 
                        help="Path to the model configuration YAML file")
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
    parser.add_argument("--mut_rate", type=float, default=0.05, help="Per-base UTR mutation probability")
    parser.add_argument("--cds_mut_rate", type=float, default=0.01, help="Per-codon CDS mutation probability")
    parser.add_argument("--min_gain", type=float, default=0.01, help="Minimum required target TE gain over WT")
    parser.add_argument("--patience", type=int, default=50, help="Number of consecutive epochs without improvement to stop training")
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
    base_model = TranslationBaseModel.from_config(args.model_config).to('cuda')
    
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
    load_result = base_model.load_pretrained_weights(args.model_weights, strict=False)
    missing_keys = getattr(load_result, "missing_keys", [])
    unexpected_keys = getattr(load_result, "unexpected_keys", [])
    head_load_issues = [
        key for key in [*missing_keys, *unexpected_keys]
        if key.startswith("heads.count.")
    ]
    if head_load_issues:
        raise RuntimeError(
            "The supplied checkpoint is incompatible with the selected head_type. "
            f"Count-head key issues: {head_load_issues}"
        )
    base_model.eval() # 显式设为评估模式，降低显存风险

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
        patience=args.patience
    )

    total_seqs = len(history_list)
    print(f"Optimization concluded. Displaying Top-K variant distribution spanning {total_seqs} sequences.")

    if total_seqs == 0:
        print(
            "Warning: No variant met the minimum target TE gain. "
            "No optimized FASTA or CSV was exported."
        )
        return

    # ==========================================================
    # High-Density Percentile Extraction & De-duplication
    # ==========================================================
    base_percentiles = [10, 30, 50]
    top_percentiles = np.linspace(70, 100, 20).tolist()
    all_percentiles = base_percentiles + top_percentiles

    selected_variants = {}
    seen_indices = set()  

    for p in all_percentiles:
        idx = min(
            int((100 - p) / 100 * (total_seqs - 1)),
            total_seqs - 1
        )
        
        if idx in seen_indices:
            continue
        seen_indices.add(idx)

        info = history_list[idx]
        p_label = f"{p:.1f}".rstrip('0').rstrip('.')

        selected_variants[f"Variant_{p_label}th_Percentile"] = {
            "sequence": info["sequence"],
            "fitness": info["fitness"],
            "target_te": info["target_te"],
            "neg_te": info["neg_te"],
            "specificity_fc": info["specificity_fc"],
            "target_te_fold_change": info["target_te_fold_change"],
            "min_acceptable_te": info["min_acceptable_te"],
        }
            
    # Data Export
    os.makedirs(args.outdir, exist_ok=True)
    
    output_fasta = os.path.join(args.outdir, f"{args.prefix}_M{args.mode}_Spec_seed{args.seed}.fasta")
    output_csv = os.path.join(args.outdir, f"{args.prefix}_M{args.mode}_Spec_seed{args.seed}.csv")
    output_png = os.path.join(args.outdir, f"{args.prefix}_M{args.mode}_Spec_seed{args.seed}_convergence.png")

    results_list = []

    with open(output_fasta, "w") as f_out:

        for label, data in selected_variants.items():

            f_out.write(
                f">{label}"
                f" | Fitness={data['fitness']:.4f}"
                f" | Target_TE={data['target_te']:.4f}"
                f" | Target_TE_FC={data['target_te_fold_change']:.4f}"
                f" | Max_Neg_TE={data['neg_te']:.4f}"
                f" | SpecFC={data['specificity_fc']:.4f}\n"
            )

            f_out.write(f"{data['sequence']}\n")

            row_dict = {
                "Variant_Name": label,
                "Fitness": data["fitness"],
                "Target_TE": data["target_te"],
                "Target_TE_Fold_Change": data["target_te_fold_change"],
                "Minimum_Acceptable_TE": data["min_acceptable_te"],
                "Max_Neg_TE": data["neg_te"],
                "Specificity_FC": data["specificity_fc"],
            }

            for neg_cell in args.neg_cells:

                neg_score = optimizer.predict_te(
                    data["sequence"],
                    cds_start_idx,
                    cds_end_idx,
                    cell_type=neg_cell
                )

                row_dict[f"Neg_TE_{neg_cell}"] = neg_score

            row_dict["Sequence"] = data["sequence"]

            results_list.append(row_dict)
            
    df_results = pd.DataFrame(results_list)
    df_results.to_csv(output_csv, index=False)

    print("Gradient Selection of Optimized Variants:")
    for res in results_list:
        neg_info = " | ".join([f"{k}: {v:.4f}" for k, v in res.items() if k.startswith("Neg_TE_")])
        print(f"[{res['Variant_Name']:<25}] Target TE: {res['Target_TE']:.4f} | {neg_info} | Max_Neg_TE: {res['Max_Neg_TE']:.4f}")

    plot_convergence(best_hist, curr_hist, wt_target_te, output_png)

    print(f"Pipeline executed successfully. Artifacts saved to: {args.outdir}")

if __name__ == "__main__":
    main()
