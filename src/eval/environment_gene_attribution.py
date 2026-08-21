"""Environmental-gene attribution for the BaseModel conditioning path."""

import numpy as np
import pandas as pd

from model.base_model import BaseModel
from utils import unwrap_model


def _unwrap(model):
    """Return and validate the exact unwrapped BaseModel instance."""
    raw = unwrap_model(model)
    if type(raw) is not BaseModel:
        raise TypeError(
            "environment_gene_attribution requires an exact "
            "model.base_model.BaseModel instance."
        )
    return raw


def _load_gene_names(gene_order_path=None, gene_annot_path="/home/user/data3/rbase/genome_ref/Homo_sapiens/hg38/ens_genes_v112.txt"):
    if gene_order_path is None:
        gene_order_path = "/home/user/data3/rbase/translation_model/models/src/config/global_anchor_gene_order.txt"
    with open(gene_order_path) as f:
        ensg_list = [line.strip() for line in f if line.strip()]

    ensg2name = {}
    if gene_annot_path is not None:
        with open(gene_annot_path) as f:
            header = f.readline().strip().split('\t')
            gid_col = header.index('Gene stable ID') if 'Gene stable ID' in header else 0
            gname_col = header.index('Gene name') if 'Gene name' in header else 2
            for line in f:
                cols = line.strip().split('\t')
                if len(cols) > max(gid_col, gname_col):
                    ensg2name[cols[gid_col]] = cols[gname_col]

    gene_names = [ensg2name.get(e, e) for e in ensg_list]
    return gene_names


def compute_adaLN_gene_attribution(
        model,
        gene_names=None,
        top_k=50,
        gene_annot_path=None,
        gene_order_path=None):
    raw = _unwrap(model)
    n_layers, d_expr = len(raw.encoder.encoder_layers), raw.d_expr
    if gene_names is None:
        gene_names = (
            _load_gene_names(gene_order_path=gene_order_path)
            if gene_annot_path is None
            else _load_gene_names(
                gene_order_path=gene_order_path,
                gene_annot_path=gene_annot_path,
            )
        )
    gene_names = list(gene_names)
    if len(gene_names) != d_expr:
        raise ValueError(
            f"gene_names length {len(gene_names)} does not match d_expr={d_expr}."
        )

    W_proj1 = raw.expr_projector[1].weight.detach().cpu().numpy()
    W_proj1_expr = np.abs(W_proj1[:, :d_expr])
    W_proj2 = np.abs(raw.expr_projector[4].weight.detach().cpu().numpy())

    all_attr = []
    for layer_idx in range(n_layers):
        for sub_idx, sub_name in [(0, 'attn'), (1, 'ffn')]:
            sublayer = raw.encoder.encoder_layers[layer_idx].sublayers[sub_idx]
            if not hasattr(sublayer, 'adaLN_modulation'):
                continue
            mod = sublayer.adaLN_modulation[1]
            W_ada = np.abs(mod.weight.detach().cpu().numpy())
            ada_imp = W_ada.sum(axis=0)
            gene_scores = ada_imp @ W_proj2 @ W_proj1_expr

            top_idx = np.argsort(gene_scores)[::-1][:top_k]
            for gi in top_idx:
                name = gene_names[gi] if gi < len(gene_names) else f"GENE_{gi}"
                all_attr.append({
                    'layer': layer_idx, 'layer_module': f'L{layer_idx}-{sub_name}',
                    'gene': name, 'gene_idx': gi, 'score': float(gene_scores[gi]),
                })

    df = pd.DataFrame(all_attr)
    if df.empty:
        return pd.DataFrame(columns=[
            'layer', 'layer_module', 'gene', 'gene_idx', 'score',
            'score_norm',
        ])
    df['score_norm'] = df.groupby('layer_module')['score'].transform(
        lambda values: (
            values / values.max()
            if values.max() > 0
            else np.zeros(len(values))
        )
    )
    return df
