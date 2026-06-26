"""
Deep-learning interpretability analysis for cell-type-specific translation regulation.

Three complementary approaches:
  1. AdaLN gene attribution — which genes in the 16k expression vector drive
     the cell-type-specific modulation (gamma/beta/alpha) at each layer.
  2. Attention map analysis — how attention patterns across sequence positions
     change between cell types, identifying putative regulatory motifs.
  3. Input saliency — gradient of predicted TE w.r.t. input one-hot sequence,
     highlighting nucleotides that most influence the prediction.

All methods use the trained model directly—no k-mer enrichment on binned transcripts.

Usage:
    from eval.interpretability_analysis import TranslationInterpretabilityAnalyzer

    analyzer = TranslationInterpretabilityAnalyzer(
        model=model,
        dataset=dataset,
        cell_expr_dict=cell_expr_dict,
        gene_names=gene_names,          # optional: list of 16k gene names
        seq_dict=seq_dict,              # optional: {tid: "ACGT..."}
        tx_cds=tx_cds,                  # optional: CDS metadata
        out_dir="./interpretability",
    )
    analyzer.run_full_analysis(n_transcripts=50, cell_types=None)
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# Helper: register hooks to capture intermediate activations
# ============================================================
class ActivationCapture:
    """Capture intermediate activations from named modules via hooks."""
    def __init__(self, model, layer_patterns=None):
        self.model = model
        self.activations = {}
        self.handles = []
        self.layer_patterns = layer_patterns or [
            'encoder.encoder_layers', 'adaLN_modulation', 'multi_headed_attention'
        ]

    def _hook_fn(self, name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self.activations[name] = output.detach().cpu()
            elif isinstance(output, (tuple, list)) and len(output) > 0:
                self.activations[name] = output[0].detach().cpu()
        return hook

    def register(self):
        for name, module in self.model.named_modules():
            if any(p in name for p in self.layer_patterns):
                self.handles.append(module.register_forward_hook(self._hook_fn(name)))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def clear(self):
        self.activations.clear()


# ============================================================
# Main Analyzer
# ============================================================
class TranslationInterpretabilityAnalyzer:
    def __init__(self, model, dataset, cell_expr_dict, 
                 gene_names=None, seq_dict=None, tx_cds=None,
                 out_dir="./interpretability", device=None):
        """
        Args:
            model: trained TranslationBaseModel
            dataset: TranslationDataset (for seq_emb, count_emb, cell_type, expr_vector)
            cell_expr_dict: {cell_type: np.array of shape (d_expr,)} for expression lookup
            gene_names: optional list of gene names matching d_expr order
            seq_dict: optional {tid: "ACGT..."} for sequence context
            tx_cds: optional {tid: {cds_start_pos, cds_end_pos, cds_frames, ...}}
            out_dir: output directory
        """
        self.model = model
        self.dataset = dataset
        self.cell_expr_dict = cell_expr_dict
        self.gene_names = gene_names
        self.seq_dict = seq_dict or {}
        self.tx_cds = tx_cds or {}
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.device = device or next(model.parameters()).device
        self.model.eval()

        self.d_expr = model.d_expr
        self.n_layers = len(model.encoder.encoder_layers)
        self.n_heads = model.n_heads

        # Captured activations
        self.adaLN_weights = {}     # layer_idx -> Linear weight (d_model*3, 32)
        self.attention_maps = {}    # (cell_type, tid) -> {layer: attn_weights}
        self.saliency_maps = {}     # (cell_type, tid) -> saliency array
        self.te_predictions = {}    # (cell_type, tid) -> float

    # ============================================================
    # 1. AdaLN gene attribution
    # ============================================================
    def _get_adaLN_weights(self):
        """Extract the trained weights of adaLN_modulation Linear layers."""
        for layer_idx in range(self.n_layers):
            layer = self.model.encoder.encoder_layers[layer_idx]
            # sublayers[0] = AddAdaZeroLayerNorm for attention
            # sublayers[1] = AddAdaZeroLayerNorm for FFN
            for sub_idx in [0, 1]:
                mod = layer.sublayers[sub_idx].adaLN_modulation[1]  # nn.Linear(32, d_model*3)
                key = f"layer{layer_idx}_sublayer{sub_idx}"
                self.adaLN_weights[key] = mod.weight.detach().cpu().numpy()  # (d_model*3, 32)

    def compute_gene_attribution(self, top_k=50):
        """
        Attribute each AdaLN module's modulation to input genes via
        the chain: gene_i -> expr_projector -> compact_style -> adaLN_modulation.

        Computes: contribution(gene_i, layer_j) = sum_over_dim(|W_adaLN|) @ |W_projector[:, i]|
        where W_projector is the first Linear in expr_projector (d_cell_env x d_expr).

        Returns DataFrame with columns: layer_module, gene, attribution_score
        """
        print("Computing AdaLN gene attribution...")

        # Extract projector weight: Linear(d_expr + d_species, d_cell_env)
        proj_linear = self.model.expr_projector[1]  # nn.Linear
        W_proj = proj_linear.weight.detach().cpu().numpy()  # (d_cell_env, d_expr + d_species)
        # Take only the gene expression part (first d_expr columns)
        W_proj_expr = np.abs(W_proj[:, :self.d_expr])  # (d_cell_env, d_expr)

        # For each adaLN layer, compute gene importance
        all_attributions = []
        self._get_adaLN_weights()

        for key, W_ada in self.adaLN_weights.items():
            # W_ada: (d_model*3, 32), take absolute sum over output dim
            ada_importance = np.abs(W_ada).sum(axis=0)  # (32,)
            # Project back through the second stage: compact_style -> adaLN
            # The compact_style dimension is 32, which is the adaptive_dim
            # We need the chain: gene -> W_proj -> d_cell_env -> nonlinear -> Linear -> 32
            # Approximate: multiply adaLN importance by the full projector chain
            # W_proj_expr: (d_cell_env, d_expr)
            # The second linear: Linear(d_cell_env, 32) — we need this too
            proj_linear2 = self.model.expr_projector[3]  # nn.Linear(d_cell_env, 32)
            W_proj2 = np.abs(proj_linear2.weight.detach().cpu().numpy())  # (32, d_cell_env)

            # Contribution of gene i to this layer's adaLN:
            # sum_{j} ada_importance[j] * sum_{k} W_proj2[j,k] * W_proj_expr[k,i]
            gene_scores = ada_importance @ W_proj2 @ W_proj_expr  # (d_expr,)

            layer_name = key.replace('layer', 'L').replace('_sublayer0', '-attn').replace('_sublayer1', '-ffn')
            for gene_idx in np.argsort(gene_scores)[::-1][:top_k]:
                gene_name = (self.gene_names[gene_idx] 
                             if self.gene_names and gene_idx < len(self.gene_names)
                             else f"GENE_{gene_idx}")
                all_attributions.append({
                    'layer_module': layer_name,
                    'gene': gene_name,
                    'gene_idx': gene_idx,
                    'attribution_score': gene_scores[gene_idx],
                })

        self.gene_attr_df = pd.DataFrame(all_attributions)
        # Normalize per layer
        self.gene_attr_df['attribution_norm'] = (
            self.gene_attr_df.groupby('layer_module')['attribution_score']
            .transform(lambda x: x / x.max())
        )

        csv_path = os.path.join(self.out_dir, "adaLN_gene_attribution.csv")
        self.gene_attr_df.to_csv(csv_path, index=False)
        print(f"Gene attribution saved to {csv_path}")

        # Top genes across all layers
        top_genes = (self.gene_attr_df.groupby('gene')['attribution_score'].sum()
                     .sort_values(ascending=False).head(top_k))
        print(f"Top {top_k} genes by summed attribution:")
        for g, s in top_genes.items():
            print(f"  {g}: {s:.4f}")

        return self.gene_attr_df

    # ============================================================
    # 2. Attention map analysis
    # ============================================================
    def _register_attention_hooks(self):
        """Register hooks to capture attention weights from all layers."""
        self._attn_activations = {}
        self._attn_handles = []

        def make_hook(layer_idx):
            def hook(module, input, output):
                # output is (bs, seq_len, d_model) after unifying heads
                # We need the raw attention weights, so we hook inside MultiHeadedAttention
                pass
            return hook

        for layer_idx, layer in enumerate(self.model.encoder.encoder_layers):
            attn = layer.multi_headed_attention

            # Hook the attention weight computation directly
            def make_attn_hook(l_idx):
                def attn_hook(module, input, output):
                    # Store the output representation — we'll compute attention
                    # externally via a separate forward pass
                    self._attn_activations[f'L{l_idx}'] = output.detach().cpu()
                return attn_hook

            self._attn_handles.append(
                attn.register_forward_hook(make_attn_hook(layer_idx))
            )

    def _remove_attention_hooks(self):
        for h in self._attn_handles:
            h.remove()
        self._attn_handles = []

    def compute_attention_maps(self, tids, cell_types, seq_len=500):
        """
        For a set of transcripts and cell types, extract the raw attention
        weights from every layer/head using a custom forward pass that
        returns attention weights explicitly.

        Stores: self.attention_maps[(tid, cell_type)] = {
            layer_idx: np.array of shape (n_heads, L, L)
        }
        """
        print(f"Computing attention maps for {len(tids)} transcripts "
              f"x {len(cell_types)} cell types...")

        # Build a modified forward that returns attention weights
        # We need to re-implement the encoder loop to capture per-layer attention

        for tid in tqdm(tids, desc="Attention maps"):
            # Find sample index in dataset
            tid_samples = []
            for i in range(len(self.dataset)):
                uuid, species, ct, ev, mi, se, ce = self.dataset[i]
                if str(uuid).rsplit('-', 2)[0] == tid:
                    tid_samples.append((i, ct, ev, se, ce, mi))
                    if len(tid_samples) >= len(set(cell_types)):
                        break

            for ct_target in cell_types:
                # Find sample with matching cell type, or use expression override
                sample = None
                for idx, ct, ev, se, ce, mi in tid_samples:
                    if ct == ct_target:
                        sample = (se, ce, ev, mi)
                        break

                if sample is None and tid_samples:
                    # Use first sample but override with target cell expression
                    _, _, _, se, ce, mi = tid_samples[0]
                    ev = torch.from_numpy(
                        self.cell_expr_dict.get(ct_target, np.zeros(self.d_expr))
                    ).float()
                    sample = (se, ce, ev, mi)

                if sample is None:
                    continue

                se, ce, ev, mi = sample
                se = se.unsqueeze(0).to(self.device)  # (1, L, 4)
                ce = ce.unsqueeze(0).to(self.device)  # (1, L, 1)
                ev = ev.unsqueeze(0).to(self.device)   # (1, d_expr)

                # Forward pass with attention extraction
                layer_maps = self._forward_with_attention(se, ce, ev, seq_len)
                if layer_maps:
                    self.attention_maps[(tid, ct_target)] = layer_maps

        n_pairs = len(self.attention_maps)
        print(f"Extracted attention maps for {n_pairs} (transcript, cell_type) pairs")

    def _forward_with_attention(self, seq_batch, count_batch, expr_vector, max_len=None):
        """
        Forward pass that returns per-layer attention weights.
        Returns dict: layer_idx -> (n_heads, L, L) numpy array.
        """
        bs = seq_batch.shape[0]
        device = seq_batch.device

        # Embed
        src_embs = self.model.src_emb(seq_batch, count_batch)
        src_mask = (seq_batch[:, :, 0] != 0).to(device)  # non-padding positions

        # Species (default to 0)
        species_idx = torch.zeros(bs, dtype=torch.long, device=device)
        species_emb = self.model.species_embedding(species_idx)

        # Expression -> compact_style
        expr_input = torch.cat([expr_vector, species_emb], dim=-1)
        compact_style = self.model.expr_projector(expr_input)  # (bs, 32)

        layer_maps = {}
        src_reps = src_embs

        for layer_idx, encoder_layer in enumerate(self.model.encoder.encoder_layers):
            # --- Attention sublayer ---
            sublayer = encoder_layer.sublayers[0]
            style = sublayer.adaLN_modulation(compact_style)
            gamma, beta, alpha = style.chunk(3, dim=-1)
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
            alpha = alpha.unsqueeze(1)

            normed = (1 + gamma) * sublayer.LN(src_reps) + beta

            # Get attention weights directly
            attn_module = encoder_layer.multi_headed_attention
            bs_, L, d_model = normed.shape
            head_dim = attn_module.head_dim
            n_heads = attn_module.heads

            q = attn_module.toqueries(normed).view(bs_, L, n_heads, head_dim).transpose(1, 2)
            k = attn_module.tokeys(normed).view(bs_, L, n_heads, head_dim).transpose(1, 2)
            v = attn_module.tovalues(normed).view(bs_, L, n_heads, head_dim).transpose(1, 2)

            # RoPE
            if hasattr(attn_module, 'RoPE'):
                q = attn_module.RoPE(q)
                k = attn_module.RoPE(k)

            # Attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(head_dim)
            mask = src_mask.unsqueeze(1).unsqueeze(2)
            scores.masked_fill_(~mask, float('-inf'))
            attn_weights = torch.softmax(scores, dim=-1)

            # Store (truncate to max_len for memory)
            L_out = min(L, max_len) if max_len else L
            layer_maps[layer_idx] = attn_weights[0, :, :L_out, :L_out].detach().cpu().numpy()

            # Complete the sublayer forward (for subsequent layers)
            attn_out = torch.matmul(attn_weights, v)
            attn_out = attn_out.transpose(1, 2).reshape(bs_, L, n_heads * head_dim)
            attn_out = attn_module.unifyheads(attn_out)
            attn_out = attn_module.dropout(attn_out)
            src_reps = src_reps + alpha * sublayer.dropout(attn_out)

            # --- FFN sublayer ---
            sublayer2 = encoder_layer.sublayers[1]
            style2 = sublayer2.adaLN_modulation(compact_style)
            gamma2, beta2, alpha2 = style2.chunk(3, dim=-1)
            gamma2 = gamma2.unsqueeze(1)
            beta2 = beta2.unsqueeze(1)
            alpha2 = alpha2.unsqueeze(1)

            normed2 = (1 + gamma2) * sublayer2.LN(src_reps) + beta2
            ffn_out = sublayer2.dropout(encoder_layer.ffn(normed2))
            src_reps = src_reps + alpha2 * ffn_out

        return layer_maps

    # ============================================================
    # 3. Input nucleotide saliency via gradient
    # ============================================================
    def compute_input_saliency(self, tids, cell_types):
        """
        For each (transcript, cell_type) pair, compute:
          saliency = |d(TE_pred)/d(one_hot_input)|
        over the CDS region. This highlights which nucleotide positions
        most influence the predicted TE for that cell type.

        Stores: self.saliency_maps[(tid, cell_type)] = {
            'saliency': (L,) array of gradient magnitudes,
            'sequence': str of CDS sequence,
            'cds_start': int (0-based),
            'cds_end': int (0-based),
        }
        """
        print(f"Computing input saliency for {len(tids)} transcripts "
              f"x {len(cell_types)} cell types...")

        from eval.calculate_te import calculate_morf_mean_signal

        for tid in tqdm(tids, desc="Saliency"):
            if tid not in self.tx_cds:
                continue
            cds_info = self.tx_cds[tid]
            cds0_start = cds_info.get('cds_start_pos', -1) - 1
            cds0_end = cds_info.get('cds_end_pos', -1) - 1
            if cds0_start < 0 or cds0_end <= cds0_start:
                continue

            seq = self.seq_dict.get(tid, '') if self.seq_dict else ''

            for ct_target in cell_types:
                # Find matching sample
                found = False
                for i in range(len(self.dataset)):
                    uuid, species, ct, ev, mi, se, ce = self.dataset[i]
                    if str(uuid).rsplit('-', 2)[0] != tid:
                        continue
                    if ct == ct_target:
                        se_sample = se.clone()
                        ce_sample = ce.clone()
                        ev_sample = ev.clone() if torch.is_tensor(ev) else torch.from_numpy(np.array(ev))
                        found = True
                        break

                if not found:
                    continue

                se_sample = se_sample.unsqueeze(0).to(self.device).requires_grad_(True)
                ce_sample = ce_sample.unsqueeze(0).to(self.device)
                ev_sample = ev_sample.unsqueeze(0).to(self.device)

                # Forward
                with torch.enable_grad():
                    out = self.model.predict(
                        seq_batch=se_sample,
                        count_batch=ce_sample,
                        expr_vector=ev_sample,
                        head_names=['count'],
                        return_numpy=False,
                    )
                    pred_profile = out['count']
                    if isinstance(pred_profile, dict):
                        pred_profile = pred_profile.get('profile', pred_profile)

                    # TE = mean frame0 signal in CDS
                    # We need a differentiable TE proxy
                    # Use mean of pred_profile over CDS region
                    te_pred = pred_profile[0, cds0_start:cds0_end:3, 0].mean()

                te_pred.backward()
                grad = se_sample.grad[0].detach().cpu().numpy()  # (L, 4)
                saliency = np.abs(grad).sum(axis=-1)  # (L,) sum over ACGT channels

                self.saliency_maps[(tid, ct_target)] = {
                    'saliency': saliency,
                    'cds_start': cds0_start,
                    'cds_end': cds0_end,
                    'sequence': seq,
                    'te_pred': float(te_pred.detach().cpu()),
                }

                se_sample.grad = None

        print(f"Computed saliency for {len(self.saliency_maps)} pairs")

    # ============================================================
    # 4. Aggregate: differentially-attended positions
    # ============================================================
    def find_differential_attention_positions(self, tid, cell_a, cell_b,
                                               layer_idx=-1, head_idx=None,
                                               top_k=20):
        """
        For a given transcript, find positions where attention patterns
        differ most between two cell types.

        Returns list of (position, diff_score, seq_context).
        """
        key_a = (tid, cell_a)
        key_b = (tid, cell_b)
        if key_a not in self.attention_maps or key_b not in self.attention_maps:
            raise ValueError("Run compute_attention_maps() first for these pairs.")

        map_a = self.attention_maps[key_a][layer_idx]  # (n_heads, L, L)
        map_b = self.attention_maps[key_b][layer_idx]

        if head_idx is not None:
            map_a = map_a[head_idx:head_idx+1]
            map_b = map_b[head_idx:head_idx+1]

        # Row-wise difference (which positions are attended to differently)
        row_diff = np.abs(map_a - map_b).mean(axis=0).mean(axis=0)  # (L,) mean over heads & query

        seq = self.seq_dict.get(tid, '')
        cds_info = self.tx_cds.get(tid, {})
        cds_start = cds_info.get('cds_start_pos', -1) - 1
        cds_end = cds_info.get('cds_end_pos', -1) - 1

        L = len(row_diff)
        top_positions = np.argsort(row_diff)[::-1][:top_k]
        results = []
        for pos in top_positions:
            ctx_start = max(0, pos - 15)
            ctx_end = min(L, pos + 16)
            region = '5UTR' if pos < cds_start else ('CDS' if pos < cds_end else '3UTR')
            results.append({
                'position': int(pos),
                'diff_score': float(row_diff[pos]),
                'region': region,
                'sequence_context': seq[ctx_start:ctx_end] if seq else 'N/A',
                'is_cds': cds_start <= pos < cds_end,
            })

        return results

    def aggregate_differential_attention(self, tids, cell_pairs, layer_idx=-1):
        """
        Aggregate differential attention across multiple transcripts and
        cell-type pairs. Returns position-wise mean diff score aligned to CDS start.
        """
        print(f"Aggregating differential attention across {len(tids)} transcripts...")
        all_diffs_aligned = []

        for tid in tids:
            if tid not in self.tx_cds:
                continue
            cds_start = self.tx_cds[tid].get('cds_start_pos', -1) - 1

            for ca, cb in cell_pairs:
                try:
                    results = self.find_differential_attention_positions(
                        tid, ca, cb, layer_idx=layer_idx, top_k=500
                    )
                except ValueError:
                    continue

                for r in results:
                    all_diffs_aligned.append({
                        'tid': tid,
                        'cell_a': ca, 'cell_b': cb,
                        'pos_from_cds_start': r['position'] - cds_start,
                        'diff_score': r['diff_score'],
                        'region': r['region'],
                    })

        if not all_diffs_aligned:
            print("No differential attention data collected.")
            return None

        self.diff_attn_df = pd.DataFrame(all_diffs_aligned)
        csv_path = os.path.join(self.out_dir, "differential_attention.csv")
        self.diff_attn_df.to_csv(csv_path, index=False)
        print(f"Differential attention saved: {len(self.diff_attn_df)} positions")
        return self.diff_attn_df

    # ============================================================
    # 5. Cell-type-specific saliency aggregation
    # ============================================================
    def aggregate_saliency_by_cell_type(self):
        """
        Aggregate saliency maps per cell type, aligned to CDS start.
        Returns DataFrame for plotting mean saliency profiles.
        """
        if not self.saliency_maps:
            print("Run compute_input_saliency() first.")
            return None

        records = []
        for (tid, ct), data in self.saliency_maps.items():
            sal = data['saliency']
            cds_start = data['cds_start']
            cds_end = data['cds_end']
            for pos in range(len(sal)):
                records.append({
                    'cell_type': ct,
                    'tid': tid,
                    'pos_from_cds_start': pos - cds_start,
                    'saliency': float(sal[pos]),
                    'region': ('5UTR' if pos < cds_start
                               else 'CDS' if pos < cds_end else '3UTR'),
                })

        self.saliency_agg_df = pd.DataFrame(records)
        csv_path = os.path.join(self.out_dir, "aggregated_saliency.csv")
        self.saliency_agg_df.to_csv(csv_path, index=False)
        print(f"Aggregated saliency: {len(self.saliency_agg_df)} entries")
        return self.saliency_agg_df

    # ============================================================
    # Full pipeline
    # ============================================================
    def run_full_analysis(self, n_transcripts=50, cell_types=None,
                          diff_cell_pairs=None, top_k_genes=50):
        """
        Run the complete interpretability pipeline.

        Args:
            n_transcripts: number of transcripts to analyze
            cell_types: list of cell types (default: all in cell_expr_dict)
            diff_cell_pairs: list of (cell_a, cell_b) tuples for differential analysis
            top_k_genes: number of top genes to report in attribution
        """
        if cell_types is None:
            cell_types = list(self.cell_expr_dict.keys())
        cell_types = cell_types[:10]  # limit for memory

        # Get top variable transcripts from dataset
        print("Selecting transcripts...")
        tids = self._select_transcripts(n_transcripts, cell_types)
        print(f"Selected {len(tids)} transcripts")

        # 1. Gene attribution
        print("\n" + "=" * 60)
        print("1. AdaLN Gene Attribution")
        print("=" * 60)
        self.compute_gene_attribution(top_k=top_k_genes)

        # 2. Attention maps
        print("\n" + "=" * 60)
        print("2. Attention Map Analysis")
        print("=" * 60)
        self.compute_attention_maps(tids[:20], cell_types[:5])

        # 3. Input saliency
        print("\n" + "=" * 60)
        print("3. Input Nucleotide Saliency")
        print("=" * 60)
        self.compute_input_saliency(tids[:10], cell_types[:3])

        # 4. Differential attention aggregation
        if diff_cell_pairs is None and len(cell_types) >= 2:
            diff_cell_pairs = [
                (cell_types[0], cell_types[1]),
                (cell_types[0], cell_types[2]) if len(cell_types) > 2 else None,
            ]
            diff_cell_pairs = [p for p in diff_cell_pairs if p is not None]

        if diff_cell_pairs and self.attention_maps:
            print("\n" + "=" * 60)
            print("4. Differential Attention Aggregation")
            print("=" * 60)
            self.aggregate_differential_attention(tids, diff_cell_pairs)

        # 5. Saliency aggregation
        if self.saliency_maps:
            print("\n" + "=" * 60)
            print("5. Saliency Aggregation")
            print("=" * 60)
            self.aggregate_saliency_by_cell_type()

        print(f"\nAnalysis complete. Results in {self.out_dir}")

    def _select_transcripts(self, n, cell_types):
        """Select transcripts that appear in the most cell types."""
        tid_cell_counts = defaultdict(set)
        for i in range(len(self.dataset)):
            uuid, species, ct, ev, mi, se, ce = self.dataset[i]
            if ct in cell_types:
                tid = str(uuid).rsplit('-', 2)[0]
                tid_cell_counts[tid].add(ct)

        # Sort by number of cell types, take top n
        sorted_tids = sorted(tid_cell_counts.items(), key=lambda x: len(x[1]), reverse=True)
        return [t for t, _ in sorted_tids[:n]]
