"""Sequence-only Transformer with standard LayerNorm for ablation studies."""

from typing import Any, Dict, List, Optional, Union

import torch

from model.base_model import BaseModel
from model.model_modules import Encoder, EncoderLayer


class TranslationBaseModel(BaseModel):
    """BaseModel-compatible sequence encoder without expression conditioning."""

    def __init__(
        self,
        d_seq: int,
        d_model: int,
        d_count: int = 1,
        n_heads: int = 8,
        number_of_layers: int = 12,
        d_ff: int = 2048,
        p_drop: float = 0.1,
        model_name: str = "base_model_LN",
    ):
        super().__init__(
            d_seq=d_seq,
            d_model=d_model,
            d_expr=1,
            d_cell_env=1,
            all_species=[],
            d_species=1,
            n_heads=n_heads,
            number_of_layers=number_of_layers,
            d_ff=d_ff,
            adaptive_dim=1,
            p_drop=p_drop,
            model_name=model_name,
        )
        self.d_count = int(d_count)
        self.encoder = Encoder(
            EncoderLayer(d_model, d_ff, n_heads, p_drop), number_of_layers
        )
        del self.species_embedding
        del self.expr_projector
        self._constructor_args = {
            "d_seq": d_seq,
            "d_count": d_count,
            "d_model": d_model,
            "n_heads": n_heads,
            "number_of_layers": number_of_layers,
            "d_ff": d_ff,
            "p_drop": p_drop,
            "model_name": model_name,
        }
        self._init_parameters()

    def forward(
        self,
        seq_batch: torch.Tensor,
        cell_type: Any = None,
        expr_vector: Optional[torch.Tensor] = None,
        species: Any = None,
        src_mask: Optional[torch.Tensor] = None,
        head_names: Optional[List[str]] = None,
        head_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
        count_batch: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if not isinstance(seq_batch, torch.Tensor) or seq_batch.dim() != 3:
            raise ValueError("seq_batch must be a tensor with shape (B, L, d_seq)")
        batch_size, seq_len, feature_dim = seq_batch.shape
        if feature_dim != self.d_seq:
            raise ValueError(f"Expected d_seq={self.d_seq}, got {feature_dim}")
        if src_mask is None:
            src_mask = (seq_batch != -1).any(dim=-1)
        else:
            src_mask = src_mask.to(device=seq_batch.device, dtype=torch.bool)
            if src_mask.shape != (batch_size, seq_len):
                raise ValueError(
                    f"src_mask shape {tuple(src_mask.shape)} != ({batch_size}, {seq_len})"
                )

        encoder_out = self.encoder(self.seq_embedding(seq_batch), src_mask)
        if not head_names:
            return encoder_out

        outputs = {}
        head_inputs = head_inputs or {}
        for name in head_names:
            if name not in self.heads:
                raise KeyError(f"Head {name!r} not found. Available: {self.list_heads()}")
            outputs[name] = self.heads[name](
                encoder_out, src_mask, **dict(head_inputs.get(name, {}))
            )
        return outputs

    @classmethod
    def from_config(cls, config: Union[Dict[str, Any], str]) -> "TranslationBaseModel":
        cfg = cls._load_config_from_file(config) if isinstance(config, str) else dict(config)
        required = {"d_seq", "d_model"}
        missing = required.difference(cfg)
        if missing:
            raise ValueError(f"Missing config keys: {sorted(missing)}")
        if cfg.get("seed") is not None:
            torch.manual_seed(int(cfg["seed"]))
        return cls(
            d_seq=int(cfg["d_seq"]),
            d_count=int(cfg.get("d_count", 1)),
            d_model=int(cfg["d_model"]),
            n_heads=int(cfg.get("n_heads", 8)),
            number_of_layers=int(cfg.get("number_of_layers", 12)),
            d_ff=int(cfg.get("d_ff", 2048)),
            p_drop=float(cfg.get("p_drop", 0.1)),
            model_name=cfg.get("model_name", "base_model_LN"),
        )


BaseModelLN = TranslationBaseModel
