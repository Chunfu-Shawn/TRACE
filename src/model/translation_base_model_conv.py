"""End-to-end sequence-only convolutional model for ablation studies."""

from typing import Any, Dict, List, Optional, Union

import torch

from model.model_modules import ConvEncoder, ConvEncoderLayer
from model.sequence_only_model import SequenceOnlyModel


class BaseModelConv(SequenceOnlyModel):
    """Residual convolutional sequence encoder without environment inputs."""

    def __init__(
        self,
        d_seq: int,
        d_model: int,
        number_of_layers: int = 12,
        d_ff: int = 2048,
        kernel_size: int = 7,
        p_drop: float = 0.1,
        model_name: str = "base_model_conv",
    ):
        super().__init__(d_seq, d_model, p_drop=p_drop, model_name=model_name)
        if number_of_layers < 1 or d_ff < 1:
            raise ValueError("number_of_layers and d_ff must be positive")
        self.kernel_size = int(kernel_size)
        self.number_of_layers = int(number_of_layers)
        self.d_ff = int(d_ff)
        self.encoder = ConvEncoder(
            ConvEncoderLayer(d_model, d_ff, kernel_size, p_drop), number_of_layers
        )
        self._constructor_args = {
            "d_seq": d_seq,
            "d_model": d_model,
            "number_of_layers": number_of_layers,
            "d_ff": d_ff,
            "kernel_size": kernel_size,
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
    def from_config(cls, config: Union[Dict[str, Any], str]) -> "BaseModelConv":
        cfg = cls._load_config_from_file(config) if isinstance(config, str) else dict(config)
        required = {"d_seq", "d_model"}
        missing = required.difference(cfg)
        if missing:
            raise ValueError(f"Missing config keys: {sorted(missing)}")
        if cfg.get("seed") is not None:
            torch.manual_seed(int(cfg["seed"]))
        return cls(
            d_seq=int(cfg["d_seq"]),
            d_model=int(cfg["d_model"]),
            number_of_layers=int(cfg.get("number_of_layers", 12)),
            d_ff=int(cfg.get("d_ff", 2048)),
            kernel_size=int(cfg.get("kernel_size", 7)),
            p_drop=float(cfg.get("p_drop", 0.1)),
            model_name=cfg.get("model_name", "base_model_conv"),
        )


TranslationBaseModel = BaseModelConv
