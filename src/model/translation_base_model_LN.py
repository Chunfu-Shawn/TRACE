"""End-to-end sequence-only Transformer for LayerNorm ablation studies."""

from typing import Any, Dict, List, Optional, Union

import torch

from model.model_modules import Encoder, EncoderLayer
from model.sequence_only_model import SequenceOnlyModel


class BaseModelLN(SequenceOnlyModel):
    """Standard pre-LayerNorm Transformer without environment conditioning."""

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
        super().__init__(d_seq, d_model, p_drop=p_drop, model_name=model_name)
        if n_heads < 1 or d_model % n_heads != 0:
            raise ValueError("n_heads must be positive and divide d_model")
        if number_of_layers < 1 or d_ff < 1:
            raise ValueError("number_of_layers and d_ff must be positive")
        self.d_count = int(d_count)
        self.n_heads = int(n_heads)
        self.number_of_layers = int(number_of_layers)
        self.d_ff = int(d_ff)
        self.encoder = Encoder(
            EncoderLayer(d_model, d_ff, n_heads, p_drop), number_of_layers
        )
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
        del cell_type, expr_vector, species, count_batch, kwargs
        sequence_embeddings, src_mask = self._embed_sequence(seq_batch, src_mask)
        encoder_out = self.encoder(sequence_embeddings, src_mask)
        return self._apply_heads(encoder_out, src_mask, head_names, head_inputs)

    @classmethod
    def from_config(cls, config: Union[Dict[str, Any], str]) -> "BaseModelLN":
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


TranslationBaseModel = BaseModelLN
