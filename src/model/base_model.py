"""
BaseModel: sequence-only encoder.

No RPF (ribosome profiling footprint) input.  Input: sequence tokens only.
Output: per-position encoder representations (B, L, d_model).

Uses AdaLN-Zero encoder conditioned on a compact expression-style vector
derived from transcriptomic profile + species embedding.
"""

import os, json, yaml
import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, List, Optional, Union

from config.model_config import ModelConfig
from model.model_modules import AdaEncoder, AdaZeroEncoderLayer

__author__ = "Chunfu Xiao"
__version__ = "2.0.0"
__email__ = "chunfushawn@gmail.com"


class BaseModel(nn.Module):
    def __init__(
        self,
        d_seq: int,
        d_model: int,
        d_expr: int = 16840,
        d_cell_env: int = 64,
        all_species: List[str] = None,
        d_species: int = 16,
        n_heads: int = 8,
        number_of_layers: int = 12,
        d_ff: int = 2048,
        adaptive_dim: int = 32,
        p_drop: float = 0.1,
        model_name: str = "base_model",
    ):
        super().__init__()
        self._constructor_args = dict(
            d_seq=d_seq, d_model=d_model, d_expr=d_expr, d_cell_env=d_cell_env,
            all_species=all_species, d_species=d_species,
            n_heads=n_heads, number_of_layers=number_of_layers,
            d_ff=d_ff, adaptive_dim=adaptive_dim, p_drop=p_drop,
            model_name=model_name,
        )
        self.model_name = model_name
        self.d_seq = d_seq
        self.d_model = d_model
        self.d_expr = d_expr
        self.d_cell_env = d_cell_env
        self.d_species = d_species
        self.n_heads = n_heads
        self.adaptive_dim = adaptive_dim
        self.p_drop = float(p_drop)

        # ----- species dictionary -----
        self.all_species = all_species if all_species else []
        self.num_species = len(self.all_species) + 1
        self.species_mapping = {sp.lower(): idx + 1 for idx, sp in enumerate(self.all_species)}

        # ----- sequence embedding -----
        self.seq_embedding = nn.Sequential(
            nn.Linear(self.d_seq, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(self.p_drop),
        )

        # ----- species embedding -----
        self.species_embedding = nn.Embedding(self.num_species, self.d_species, padding_idx=0)

        # ----- expression + species → compact style projector -----
        self.expr_projector = nn.Sequential(
            nn.Dropout(min(self.p_drop * 1.5, 0.7)),
            nn.Linear(self.d_expr + self.d_species, self.d_cell_env, bias=False),
            nn.LayerNorm(self.d_cell_env),
            nn.GELU(),
            nn.Linear(self.d_cell_env, self.adaptive_dim),
        )

        # ----- encoder -----
        encoder_layer = AdaZeroEncoderLayer(self.d_model, d_ff, n_heads, self.p_drop, self.adaptive_dim)
        self.encoder = AdaEncoder(encoder_layer, number_of_layers)

        # ----- pluggable heads -----
        self.heads = nn.ModuleDict()

        self._init_parameters()
        self.register_buffer("mean_expr_vector", torch.zeros(self.d_expr))
        self.cell_expr_dict = {}

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.LayerNorm):
                if m.elementwise_affine:
                    if m.weight is not None:
                        nn.init.ones_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    @property
    def device(self):
        return self.mean_expr_vector.device

    def _default_map_location(self):
        """Return the device that currently owns this model's parameters."""
        return self.device

    # ------------------------------------------------------------------
    # Expression dictionary
    # ------------------------------------------------------------------
    def load_expression_dict(self, expr_dict: Dict[str, Union[torch.Tensor, np.ndarray, list]]):
        self.cell_expr_dict = {}
        all_vectors = []
        for cell_name, vec in expr_dict.items():
            tensor_vec = torch.as_tensor(vec, dtype=torch.float32).view(-1)
            if tensor_vec.shape[-1] != self.d_expr:
                raise ValueError(f"Vector for {cell_name} has wrong dim {tensor_vec.shape[-1]}, expected {self.d_expr}")
            self.cell_expr_dict[cell_name] = tensor_vec
            all_vectors.append(tensor_vec)
        if all_vectors:
            self.mean_expr_vector.copy_(torch.stack(all_vectors).mean(dim=0))
            print(f"[BaseModel] Loaded {len(self.cell_expr_dict)} expression profiles.")
        else:
            print("[BaseModel] Warning: empty expression dict, mean stays at zero.")

    # ------------------------------------------------------------------
    # Species normalization
    # ------------------------------------------------------------------
    def _normalize_species(self, species: Any, batch_size: int) -> torch.LongTensor:
        def _to_idx(val):
            if isinstance(val, str):
                return self.species_mapping.get(val.lower(), 0)
            try:
                ival = int(val)
                if 0 <= ival < self.num_species:
                    return ival
            except Exception:
                pass
            return 0

        if species is None:
            return torch.zeros(batch_size, dtype=torch.long)
        if isinstance(species, str):
            idx = _to_idx(species)
            return torch.full((batch_size,), idx, dtype=torch.long)
        if isinstance(species, (list, tuple, np.ndarray)):
            if len(species) == 1:
                val = species[0] if isinstance(species, (list, tuple)) else species.item()
                return torch.full((batch_size,), _to_idx(val), dtype=torch.long)
            if len(species) == batch_size:
                return torch.tensor([_to_idx(x) for x in species], dtype=torch.long)
            raise ValueError(f"species length {len(species)} != batch_size {batch_size}")
        if isinstance(species, torch.Tensor):
            if species.numel() == 1:
                return torch.full((batch_size,), _to_idx(species.item()), dtype=torch.long)
            if species.dim() == 1 and species.numel() == batch_size:
                return species.to(dtype=torch.long).clamp(0, self.num_species - 1)
        return torch.full((batch_size,), _to_idx(species), dtype=torch.long)

    # ------------------------------------------------------------------
    # Expression resolution
    # ------------------------------------------------------------------
    def _resolve_expr_vector(self, cell_type: Any, expr_vector: Any, batch_size: int) -> torch.Tensor:
        if expr_vector is not None:
            t = torch.as_tensor(expr_vector, dtype=torch.float32)
            if t.dim() == 1:
                return t.unsqueeze(0).expand(batch_size, -1).clone()
            if t.dim() == 2:
                if t.shape != (batch_size, self.d_expr):
                    raise ValueError(f"expr_vector shape {tuple(t.shape)} != ({batch_size}, {self.d_expr})")
                return t
            raise ValueError("Unsupported expr_vector shape")

        out = torch.empty(batch_size, self.d_expr, dtype=torch.float32, device=self.mean_expr_vector.device)

        def _get(name):
            if isinstance(name, str) and name in self.cell_expr_dict:
                return self.cell_expr_dict[name].to(self.mean_expr_vector.device)
            return self.mean_expr_vector

        if cell_type is None:
            out[:] = self.mean_expr_vector
        elif isinstance(cell_type, str):
            out[:] = _get(cell_type)
        elif isinstance(cell_type, (list, tuple, np.ndarray)):
            if len(cell_type) == 1:
                key = cell_type[0] if isinstance(cell_type, (list, tuple)) else cell_type.item()
                out[:] = _get(key)
            elif len(cell_type) == batch_size:
                for i, ct in enumerate(cell_type):
                    out[i] = _get(ct)
            else:
                raise ValueError(f"cell_type len {len(cell_type)} != batch_size {batch_size}")
        else:
            out[:] = self.mean_expr_vector
        return out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        seq_batch: torch.Tensor,
        cell_type: Any = None,
        expr_vector: Optional[torch.Tensor] = None,
        species: Any = None,
        src_mask: Optional[torch.Tensor] = None,
        head_names: Optional[List[str]] = None,
        head_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        seq_batch : (B, L, d_seq)
            Continuous sequence features per position.
        cell_type / expr_vector / species / src_mask : same as TranslationBaseModel.
        head_names : list of str, optional
            Names of registered heads to run.  If None/empty, returns encoder_out.
        head_inputs : dict, optional
            Per-head extra kwargs: ``{head_name: {kw: val, ...}}``.

        Returns
        -------
        encoder_out : (B, L, d_model) when no heads requested.
        outputs : dict of Tensors when head_names given.
        """
        if seq_batch.dim() != 3:
            raise ValueError(f"seq_batch must be (B, L, d_seq), got {tuple(seq_batch.shape)}")
        B, L, _ = seq_batch.shape

        # sequence embedding
        seq_embs = self.seq_embedding(seq_batch)  # (B, L, d_model)

        # expression + species → compact style
        final_expr = self._resolve_expr_vector(cell_type, expr_vector, B).to(seq_batch.device)
        species_idx = self._normalize_species(species, B).to(seq_batch.device)
        species_embs = self.species_embedding(species_idx)
        combined = torch.cat([final_expr, species_embs], dim=-1)
        compact_style = self.expr_projector(combined)  # (B, adaptive_dim)

        # src_mask validation
        if src_mask is not None:
            if src_mask.dim() != 2 or src_mask.shape[0] != B or src_mask.shape[1] != L:
                raise ValueError(f"src_mask shape {tuple(src_mask.shape)} != ({B}, {L})")
            if src_mask.dtype != torch.bool:
                src_mask = src_mask.bool()

        # encoder
        encoder_out = self.encoder(seq_embs, src_mask, compact_style)  # (B, L, d_model)

        # if no heads requested, return raw representations
        if not head_names:
            return encoder_out

        # run requested heads
        outputs = {}
        head_inputs = head_inputs or {}
        for name in head_names:
            if name not in self.heads:
                raise KeyError(f"Head '{name}' not found. Available: {list(self.heads.keys())}")
            head = self.heads[name]
            per_head_kwargs = dict(head_inputs.get(name, {}))
            outputs[name] = head(encoder_out, src_mask, **per_head_kwargs)

        return outputs

    # ------------------------------------------------------------------
    # Flexible inference
    # ------------------------------------------------------------------
    @staticmethod
    def encode_sequence(seq):
        """Convert RNA string(s) to one-hot numpy array(s).

        str -> (L, 4)
        list/tuple of str -> (B, L_max, 4)  zero-padded to max length
        """
        nt = {"A":0,"C":1,"G":2,"T":3,"U":3}

        if isinstance(seq, str):
            if not seq:
                raise ValueError("RNA sequence must not be empty")
            idx = [nt.get(c.upper(),4) for c in seq]
            return np.eye(5,dtype=np.float32)[idx,:4]

        if isinstance(seq, (list,tuple)):
            if not seq:
                raise ValueError("Expected a non-empty list or tuple of sequences")
            encoded = [BaseModel.encode_sequence(s) for s in seq]
            max_len = max(e.shape[0] for e in encoded)
            padded = np.zeros((len(encoded),max_len,4),dtype=np.float32)
            for i,e in enumerate(encoded):
                padded[i,:e.shape[0],:] = e
            return padded

        raise TypeError(f"Expected str or list/tuple of str, got {type(seq)}")


    @torch.no_grad()
    def predict(
        self,
        seq_batch: Union[torch.Tensor, np.ndarray, list],
        species: Any = None,
        cell_type: Any = None,
        expr_vector: Any = None,
        src_mask: Optional[Union[torch.Tensor, np.ndarray, list]] = None,
        head_names: Optional[List[str]] = None,
        head_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
        move_inputs_to_device: bool = True,
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, Dict, np.ndarray]:
        """
        Flexible single-sample / batched inference.

        Returns encoder_out or head outputs (numpy if return_numpy=True).
        """
        self.eval()
        dev = self.device

        raw_sequence_lengths = None
        if isinstance(seq_batch, (list, tuple)) and seq_batch and all(
            isinstance(sequence, str) for sequence in seq_batch
        ):
            raw_sequence_lengths = [len(sequence) for sequence in seq_batch]

        if isinstance(seq_batch, str) or raw_sequence_lengths is not None:
            seq_batch = self.encode_sequence(seq_batch)
        if isinstance(seq_batch, (list, tuple, np.ndarray)):
            seq_batch = torch.as_tensor(seq_batch, dtype=torch.float32)
        elif isinstance(seq_batch, torch.Tensor):
            seq_batch = seq_batch.to(dtype=torch.float32)
        if seq_batch.dim() == 2:
            seq_batch = seq_batch.unsqueeze(0)
            was_squeezed = True
        else:
            was_squeezed = False

        if src_mask is not None:
            if not isinstance(src_mask, torch.Tensor):
                src_mask = torch.as_tensor(src_mask, dtype=torch.bool)
            if src_mask.dim() == 1:
                src_mask = src_mask.unsqueeze(0)
        elif raw_sequence_lengths is not None:
            positions = torch.arange(seq_batch.shape[1]).unsqueeze(0)
            lengths = torch.tensor(raw_sequence_lengths).unsqueeze(1)
            src_mask = positions < lengths

        if move_inputs_to_device:
            seq_batch = seq_batch.to(dev)
            if src_mask is not None:
                src_mask = src_mask.to(dev)

        B = seq_batch.shape[0]
        final_expr = self._resolve_expr_vector(cell_type, expr_vector, B)

        result = self.forward(
            seq_batch=seq_batch,
            expr_vector=final_expr,
            species=species,
            src_mask=src_mask,
            head_names=head_names,
            head_inputs=head_inputs,
        )

        def _squeeze(obj):
            if isinstance(obj, torch.Tensor):
                return obj.squeeze(0) if was_squeezed and obj.shape[0] == 1 else obj
            if isinstance(obj, dict):
                return {k: _squeeze(v) for k, v in obj.items()}
            return obj

        result = _squeeze(result)

        if return_numpy:
            if isinstance(result, torch.Tensor):
                return result.cpu().numpy()
            return {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in result.items()}
        return result

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------
    @classmethod
    def _load_config_from_file(cls, path: str) -> Dict[str, Any]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        lower = path.lower()
        if lower.endswith(".json"):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        if lower.endswith((".yaml", ".yml")):
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        try:
            return json.loads(text)
        except Exception:
            return yaml.safe_load(text)

    @classmethod
    def _normalize_config(cls, config: Union[Dict, ModelConfig, str]) -> ModelConfig:
        if isinstance(config, ModelConfig):
            return config
        if isinstance(config, str):
            cfg_dict = cls._load_config_from_file(config)
        elif isinstance(config, dict):
            cfg_dict = config
        else:
            raise TypeError("config must be dict, ModelConfig, or file path")
        required = {"d_seq", "d_model", "d_expr"}
        missing = required - cfg_dict.keys()
        if missing:
            raise ValueError(f"Missing config keys: {missing}")
        return ModelConfig(**{k: cfg_dict[k] for k in cfg_dict if k in ModelConfig.__dataclass_fields__})

    @classmethod
    def from_config(cls, config: Union[Dict, ModelConfig, str]) -> "BaseModel":
        cfg = cls._normalize_config(config)
        if cfg.seed is not None:
            torch.manual_seed(int(cfg.seed))
        print(f"[BaseModel] Creating from config: {cfg}")

        model = cls(
            d_seq=cfg.d_seq, d_model=cfg.d_model, d_expr=cfg.d_expr,
            d_cell_env=cfg.d_cell_env, d_species=cfg.d_species,
            all_species=cfg.all_species, n_heads=cfg.n_heads,
            number_of_layers=cfg.number_of_layers, d_ff=cfg.d_ff,
            adaptive_dim=cfg.adaptive_dim, p_drop=cfg.p_drop,
            model_name=cfg.model_name,
        )
        if cfg.expr_dict_path is not None and os.path.isfile(cfg.expr_dict_path):
            print(f"[BaseModel] Auto-loading expression dict from {cfg.expr_dict_path}")
            if cfg.expr_dict_path.endswith(".json"):
                with open(cfg.expr_dict_path) as f:
                    expr_dict = json.load(f)
            elif cfg.expr_dict_path.endswith((".pkl", ".pickle")):
                import pickle
                with open(cfg.expr_dict_path, "rb") as f:
                    expr_dict = pickle.load(f)
            elif cfg.expr_dict_path.endswith(".pt"):
                expr_dict = torch.load(cfg.expr_dict_path, map_location="cpu")
            else:
                raise ValueError("Unsupported expr_dict_path format")
            model.load_expression_dict(expr_dict)
        return model

    def save_config(self, path: str, as_yaml: bool = False):
        cfg = dict(self._constructor_args)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if as_yaml or path.lower().endswith((".yaml", ".yml")):
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)


    # ------------------------------------------------------------------
    # Head management
    # ------------------------------------------------------------------
    def add_head(self, name: str, head_module: nn.Module, overwrite: bool = False,
                 move_to_model_device: bool = True) -> None:
        """Register a head module.  Must have a ``name`` attribute."""
        if (name in self.heads) and (not overwrite):
            raise KeyError(f"Head '{name}' exists. Use overwrite=True.")
        if move_to_model_device:
            head_module.to(self.device)
        head_name = getattr(head_module, "name", name)
        self.heads[name] = head_module
        self.model_name = f'{self.model_name}-{head_name}'

    def remove_head(self, name: str) -> None:
        """Remove a head and rebuild model_name safely."""
        if name not in self.heads:
            raise KeyError(f"Head '{name}' does not exist.")
        head = self.heads[name]
        head_name = getattr(head, "name", name)
        parts = self.model_name.split("-")
        self.model_name = "-".join([p for p in parts if p != head_name])
        del self.heads[name]

    def list_heads(self) -> List[str]:
        return list(self.heads.keys())

    def save_head(self, name: str, path: str) -> None:
        if name not in self.heads:
            raise KeyError(name)
        torch.save(self.heads[name].state_dict(), path)

    def load_head(self, name: str, path: str, map_location: Optional[str] = None) -> None:
        if name not in self.heads:
            raise KeyError(f"Head '{name}' does not exist. Register it via add_head() first.")
        map_loc = map_location or self._default_map_location()
        state = torch.load(path, map_location=map_loc)
        state = self._strip_head_module_prefix(state)
        self.heads[name].load_state_dict(state)

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------
    @staticmethod
    def _strip_head_module_prefix(state_dict):
        """Backward compat: strip ``.module.`` from old HeadAdapter checkpoints."""
        return {k.replace(".module.", "."): v for k, v in state_dict.items()}

    def load_pretrained_weights(self, ckpt_path: Optional[str], strict: bool = False, map_location: Optional[str] = None):
        if ckpt_path is None:
            return None
        map_loc = map_location or self._default_map_location()
        ckpt = torch.load(ckpt_path, map_location=map_loc)
        if isinstance(ckpt, dict) and "model" in ckpt:
            sd = ckpt["model"]
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            sd = ckpt
        else:
            raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")
        sd = self._strip_head_module_prefix(sd)
        target = self.module if hasattr(self, "module") else self
        res = target.load_state_dict(sd, strict=strict)
        print(f"[BaseModel] load_pretrained_weights: {ckpt_path} strict={strict} "
              f"missing={getattr(res,'missing_keys',None)} unexpected={getattr(res,'unexpected_keys',None)}")
        return res

    def load_lora_and_heads(self, ckpt_path: Optional[str], strict: bool = False,
                            map_location: Optional[str] = None):
        """Load a partial checkpoint (LoRA adapters + heads only)."""
        if ckpt_path is None:
            return None
        map_loc = map_location or self._default_map_location()
        ckpt = torch.load(ckpt_path, map_location=map_loc)
        if isinstance(ckpt, dict) and "model" in ckpt:
            sd = ckpt["model"]
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            sd = ckpt
        else:
            raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")
        sd = self._strip_head_module_prefix(sd)
        target = self.module if hasattr(self, "module") else self
        res = target.load_state_dict(sd, strict=strict)
        print(f"[BaseModel] load_lora_and_heads: {ckpt_path} strict={strict} "
              f"missing={getattr(res,'missing_keys',None)} unexpected={getattr(res,'unexpected_keys',None)}")
        return res
