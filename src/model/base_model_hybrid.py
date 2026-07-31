"""Configurable hybrid TRACE encoder with Pre-LN followed by Pre-AdaLN layers."""

import copy
import json
import os
import pickle
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from model.base_model import BaseModel
from model.model_modules import AdaZeroEncoderLayer, EncoderLayer


class HybridEncoder(nn.Module):
    """Run standard Pre-LN layers before expression-conditioned Pre-AdaLN layers."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_heads: int,
        p_drop: float,
        adaptive_dim: int,
        number_of_layers: int,
        pre_ln_layers: int,
        adaln_modulation_bounds: Optional[Dict[str, float]],
    ):
        super().__init__()
        if not 0 <= pre_ln_layers < number_of_layers:
            raise ValueError(
                "pre_ln_layers must be non-negative and smaller than "
                "number_of_layers"
            )

        pre_ln_layer = EncoderLayer(d_model, d_ff, n_heads, p_drop)
        pre_adaln_layer = AdaZeroEncoderLayer(
            d_model,
            d_ff,
            n_heads,
            p_drop,
            adaptive_dim,
            adaln_modulation_bounds=adaln_modulation_bounds,
        )
        adaptive_layers = number_of_layers - pre_ln_layers
        self.pre_ln_layers = int(pre_ln_layers)
        self.encoder_layers = nn.ModuleList(
            [copy.deepcopy(pre_ln_layer) for _ in range(pre_ln_layers)]
            + [copy.deepcopy(pre_adaln_layer) for _ in range(adaptive_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        src_embs: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        compact_style: torch.Tensor,
    ) -> torch.Tensor:
        """Encode sequence features and introduce conditioning only in later layers."""
        src_reps = src_embs
        for layer_index, encoder_layer in enumerate(self.encoder_layers):
            if layer_index < self.pre_ln_layers:
                src_reps = encoder_layer(src_reps, src_mask)
            else:
                src_reps = encoder_layer(src_reps, src_mask, compact_style)
        return self.norm(src_reps)


class BaseModelHybrid(BaseModel):
    """BaseModel with sequence Pre-LN followed by conditional Pre-AdaLN layers."""

    def __init__(
        self,
        d_seq: int,
        d_model: int,
        d_expr: int = 16840,
        d_cell_env: int = 64,
        all_species: Optional[List[str]] = None,
        d_species: int = 16,
        n_heads: int = 8,
        number_of_layers: int = 12,
        d_ff: int = 2048,
        adaptive_dim: int = 32,
        p_drop: float = 0.1,
        model_name: str = "base_model_hybrid_7preln_5preadaln",
        adaln_modulation_bounds: Optional[Dict[str, float]] = None,
        number_of_adaln_layers: int = 5,
    ):
        if not 1 <= number_of_adaln_layers <= number_of_layers:
            raise ValueError(
                "number_of_adaln_layers must be between 1 and number_of_layers"
            )
        super().__init__(
            d_seq=d_seq,
            d_model=d_model,
            d_expr=d_expr,
            d_cell_env=d_cell_env,
            all_species=all_species,
            d_species=d_species,
            n_heads=n_heads,
            number_of_layers=number_of_layers,
            d_ff=d_ff,
            adaptive_dim=adaptive_dim,
            p_drop=p_drop,
            model_name=model_name,
            adaln_modulation_bounds=adaln_modulation_bounds,
        )
        self.pre_adaln_layers = int(number_of_adaln_layers)
        self.pre_ln_layers = number_of_layers - self.pre_adaln_layers
        self.encoder = HybridEncoder(
            d_model=self.d_model,
            d_ff=d_ff,
            n_heads=self.n_heads,
            p_drop=self.p_drop,
            adaptive_dim=self.adaptive_dim,
            number_of_layers=number_of_layers,
            pre_ln_layers=self.pre_ln_layers,
            adaln_modulation_bounds=self.adaln_modulation_bounds,
        )
        self._initialize_hybrid_encoder()
        self._constructor_args["number_of_adaln_layers"] = self.pre_adaln_layers

    @classmethod
    def from_config(cls, config):
        """Create a hybrid model and honor number_of_adaln_layers from config."""
        cfg = cls._normalize_config(config)
        if cfg.seed is not None:
            torch.manual_seed(int(cfg.seed))
        adaln_layers = (
            5
            if cfg.number_of_adaln_layers is None
            else int(cfg.number_of_adaln_layers)
        )
        print(f"[BaseModelHybrid] Creating from config: {cfg}")
        model = cls(
            d_seq=cfg.d_seq,
            d_model=cfg.d_model,
            d_expr=cfg.d_expr,
            d_cell_env=cfg.d_cell_env,
            d_species=cfg.d_species,
            all_species=cfg.all_species,
            n_heads=cfg.n_heads,
            number_of_layers=cfg.number_of_layers,
            d_ff=cfg.d_ff,
            adaptive_dim=cfg.adaptive_dim,
            p_drop=cfg.p_drop,
            adaln_modulation_bounds=cfg.adaln_modulation_bounds,
            number_of_adaln_layers=adaln_layers,
            model_name=cfg.model_name or "base_model_hybrid_7preln_5preadaln",
        )
        if cfg.expr_dict_path is not None and os.path.isfile(cfg.expr_dict_path):
            if cfg.expr_dict_path.endswith(".json"):
                with open(cfg.expr_dict_path, encoding="utf-8") as stream:
                    expression_dict = json.load(stream)
            elif cfg.expr_dict_path.endswith((".pkl", ".pickle")):
                with open(cfg.expr_dict_path, "rb") as stream:
                    expression_dict = pickle.load(stream)
            elif cfg.expr_dict_path.endswith(".pt"):
                expression_dict = torch.load(cfg.expr_dict_path, map_location="cpu")
            else:
                raise ValueError("Unsupported expr_dict_path format")
            model.load_expression_dict(expression_dict)
        return model

    def _initialize_hybrid_encoder(self) -> None:
        """Initialize the replacement encoder and restore AdaLN-Zero gates."""
        for module in self.encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.LayerNorm) and module.elementwise_affine:
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        for module in self.encoder.modules():
            if isinstance(module, AdaZeroEncoderLayer):
                module.reset_ada_zero_parameters()
