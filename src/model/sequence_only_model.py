"""Shared infrastructure for sequence-only TRACE ablation models."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn


class SequenceOnlyModel(nn.Module):
    """Base class whose predictions depend only on sequence and padding masks.

    Environment-related arguments remain in ``forward`` and ``predict`` signatures
    solely so these models can use the same Trainer and evaluation utilities as
    BaseModel. They are never encoded or used in the computation graph.
    """

    def __init__(
        self,
        d_seq: int,
        d_model: int,
        p_drop: float = 0.1,
        model_name: str = "sequence_only_model",
    ) -> None:
        super().__init__()
        if d_seq < 1 or d_model < 1:
            raise ValueError("d_seq and d_model must be positive")
        if not 0.0 <= float(p_drop) < 1.0:
            raise ValueError("p_drop must be in [0, 1)")

        self.d_seq = int(d_seq)
        self.d_model = int(d_model)
        self.p_drop = float(p_drop)
        self.model_name = str(model_name)
        self.seq_embedding = nn.Sequential(
            nn.Linear(self.d_seq, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(self.p_drop),
        )
        self.heads = nn.ModuleDict()
        self._constructor_args: Dict[str, Any] = {}

    @property
    def device(self) -> torch.device:
        """Return the device owning the model parameters."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _default_map_location(self) -> torch.device:
        """Return a checkpoint map location derived from the current model."""
        return self.device

    def _init_parameters(self) -> None:
        """Match BaseModel initialization for shared layer types."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm) and module.elementwise_affine:
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _embed_sequence(
        self,
        seq_batch: torch.Tensor,
        src_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate sequence inputs, resolve the padding mask, and embed tokens."""
        if not isinstance(seq_batch, torch.Tensor) or seq_batch.ndim != 3:
            raise ValueError("seq_batch must be a tensor with shape (B, L, d_seq)")
        batch_size, seq_len, feature_dim = seq_batch.shape
        if feature_dim != self.d_seq:
            raise ValueError(f"Expected d_seq={self.d_seq}, got {feature_dim}")

        if src_mask is None:
            src_mask = (seq_batch != -1).any(dim=-1)
        else:
            src_mask = torch.as_tensor(
                src_mask, dtype=torch.bool, device=seq_batch.device
            )
            if src_mask.shape != (batch_size, seq_len):
                raise ValueError(
                    f"src_mask shape {tuple(src_mask.shape)} != "
                    f"({batch_size}, {seq_len})"
                )
        return self.seq_embedding(seq_batch), src_mask

    def _apply_heads(
        self,
        encoder_out: torch.Tensor,
        src_mask: torch.Tensor,
        head_names: Optional[List[str]],
        head_inputs: Optional[Dict[str, Dict[str, Any]]],
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run requested prediction heads or return encoder representations."""
        if not head_names:
            return encoder_out

        outputs: Dict[str, torch.Tensor] = {}
        inputs_by_head = head_inputs or {}
        for name in head_names:
            if name not in self.heads:
                raise KeyError(
                    f"Head {name!r} not found. Available: {self.list_heads()}"
                )
            outputs[name] = self.heads[name](
                encoder_out,
                src_mask,
                **dict(inputs_by_head.get(name, {})),
            )
        return outputs

    @staticmethod
    def encode_sequence(sequence):
        """Convert one RNA sequence or a sequence list to one-hot features."""
        nucleotide_to_index = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}

        if isinstance(sequence, str):
            if not sequence:
                raise ValueError("RNA sequence must not be empty")
            indices = [nucleotide_to_index.get(base.upper(), 4) for base in sequence]
            return np.eye(5, dtype=np.float32)[indices, :4]

        if isinstance(sequence, (list, tuple)):
            if not sequence:
                raise ValueError("Expected a non-empty sequence list")
            encoded = [SequenceOnlyModel.encode_sequence(item) for item in sequence]
            max_length = max(item.shape[0] for item in encoded)
            padded = np.zeros((len(encoded), max_length, 4), dtype=np.float32)
            for index, item in enumerate(encoded):
                padded[index, : item.shape[0]] = item
            return padded

        raise TypeError(
            f"Expected a sequence string or list/tuple, got {type(sequence)}"
        )

    @torch.no_grad()
    def predict(
        self,
        seq_batch: Union[str, torch.Tensor, np.ndarray, list],
        species: Any = None,
        cell_type: Any = None,
        expr_vector: Any = None,
        src_mask: Optional[Union[torch.Tensor, np.ndarray, list]] = None,
        head_names: Optional[List[str]] = None,
        head_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
        move_inputs_to_device: bool = True,
        return_numpy: bool = False,
        count_batch: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor], np.ndarray, Dict[str, np.ndarray]]:
        """Run sequence-only inference with BaseModel-compatible arguments."""
        del species, cell_type, expr_vector, count_batch, kwargs
        self.eval()

        raw_lengths = None
        if isinstance(seq_batch, (list, tuple)) and seq_batch and all(
            isinstance(sequence, str) for sequence in seq_batch
        ):
            raw_lengths = [len(sequence) for sequence in seq_batch]

        if isinstance(seq_batch, str) or raw_lengths is not None:
            seq_batch = self.encode_sequence(seq_batch)
        if isinstance(seq_batch, (list, tuple, np.ndarray)):
            seq_batch = torch.as_tensor(seq_batch, dtype=torch.float32)
        elif isinstance(seq_batch, torch.Tensor):
            seq_batch = seq_batch.to(dtype=torch.float32)
        else:
            raise TypeError(f"Unsupported seq_batch type: {type(seq_batch)}")

        if seq_batch.ndim == 2:
            seq_batch = seq_batch.unsqueeze(0)
            was_squeezed = True
        elif seq_batch.ndim == 3:
            was_squeezed = False
        else:
            raise ValueError(
                f"seq_batch must have 2 or 3 dimensions, got {seq_batch.ndim}"
            )

        if src_mask is not None:
            src_mask = torch.as_tensor(src_mask, dtype=torch.bool)
            if src_mask.ndim == 1:
                src_mask = src_mask.unsqueeze(0)
        elif raw_lengths is not None:
            positions = torch.arange(seq_batch.shape[1]).unsqueeze(0)
            lengths = torch.tensor(raw_lengths).unsqueeze(1)
            src_mask = positions < lengths

        if move_inputs_to_device:
            seq_batch = seq_batch.to(self.device)
            if src_mask is not None:
                src_mask = src_mask.to(self.device)

        result = self.forward(
            seq_batch=seq_batch,
            src_mask=src_mask,
            head_names=head_names,
            head_inputs=head_inputs,
        )

        def squeeze_output(value):
            if isinstance(value, torch.Tensor):
                if was_squeezed and value.shape[0] == 1:
                    return value.squeeze(0)
                return value
            if isinstance(value, dict):
                return {key: squeeze_output(item) for key, item in value.items()}
            return value

        result = squeeze_output(result)
        if not return_numpy:
            return result
        if isinstance(result, torch.Tensor):
            return result.cpu().numpy()
        return {
            key: value.cpu().numpy() if isinstance(value, torch.Tensor) else value
            for key, value in result.items()
        }

    @staticmethod
    def _load_config_from_file(path: str) -> Dict[str, Any]:
        """Load a JSON or YAML model configuration."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        lower_path = path.lower()
        if lower_path.endswith(".json"):
            with open(path, encoding="utf-8") as handle:
                return json.load(handle)
        if lower_path.endswith((".yaml", ".yml")):
            try:
                import yaml
            except ImportError as exc:
                raise ImportError("PyYAML is required to load YAML model configs") from exc
            with open(path, encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle)
            if not isinstance(loaded, dict):
                raise ValueError("Model configuration must contain a mapping")
            return loaded
        raise ValueError("Model config must use .json, .yaml, or .yml")

    def save_config(self, path: str, as_yaml: bool = False) -> None:
        """Save constructor arguments as JSON or YAML."""
        config = dict(self._constructor_args)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if as_yaml or path.lower().endswith((".yaml", ".yml")):
            try:
                import yaml
            except ImportError as exc:
                raise ImportError("PyYAML is required to save YAML model configs") from exc
            with open(path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle)
        else:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(config, handle, indent=2)

    def add_head(
        self,
        name: str,
        head_module: nn.Module,
        overwrite: bool = False,
        move_to_model_device: bool = True,
    ) -> None:
        """Register a prediction head using the BaseModel head contract."""
        if name in self.heads and not overwrite:
            raise KeyError(f"Head {name!r} already exists")
        if move_to_model_device:
            head_module.to(self.device)
        head_name = getattr(head_module, "name", name)
        self.heads[name] = head_module
        self.model_name = f"{self.model_name}-{head_name}"

    def remove_head(self, name: str) -> None:
        """Remove a registered prediction head."""
        if name not in self.heads:
            raise KeyError(f"Head {name!r} does not exist")
        head_name = getattr(self.heads[name], "name", name)
        self.model_name = "-".join(
            part for part in self.model_name.split("-") if part != head_name
        )
        del self.heads[name]

    def list_heads(self) -> List[str]:
        """Return registered prediction-head names."""
        return list(self.heads.keys())

    def save_head(self, name: str, path: str) -> None:
        """Save one prediction head state dictionary."""
        if name not in self.heads:
            raise KeyError(name)
        torch.save(self.heads[name].state_dict(), path)

    def load_head(
        self,
        name: str,
        path: str,
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Load one prediction head state dictionary."""
        if name not in self.heads:
            raise KeyError(f"Head {name!r} does not exist")
        state = torch.load(path, map_location=map_location or self.device)
        self.heads[name].load_state_dict(self._strip_head_module_prefix(state))

    @staticmethod
    def _strip_head_module_prefix(state_dict):
        """Strip legacy HeadAdapter module prefixes."""
        return {key.replace(".module.", "."): value for key, value in state_dict.items()}

    def load_pretrained_weights(
        self,
        checkpoint_path: Optional[str],
        strict: bool = False,
        map_location: Optional[Union[str, torch.device]] = None,
    ):
        """Load a full Trainer or raw state-dictionary checkpoint."""
        if checkpoint_path is None:
            return None
        checkpoint = torch.load(
            checkpoint_path, map_location=map_location or self._default_map_location()
        )
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state = checkpoint["model"]
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and all(
            isinstance(value, torch.Tensor) for value in checkpoint.values()
        ):
            state = checkpoint
        else:
            raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
        result = self.load_state_dict(
            self._strip_head_module_prefix(state), strict=strict
        )
        print(
            f"[SequenceOnlyModel] Loaded {checkpoint_path} strict={strict} "
            f"missing={result.missing_keys} unexpected={result.unexpected_keys}"
        )
        return result
