"""Model architecture facts the planner needs, read from a Hugging Face config.json."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4, "fp8": 1, "int8": 1}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    tied_embeddings: bool
    dtype_bytes: int = 2
    max_position: int = 32768

    # ---- parameter counts (dense decoder: attention + gated MLP) ----
    @property
    def embed_params(self) -> int:
        return self.vocab * self.hidden

    @property
    def layer_params(self) -> int:
        q = self.hidden * self.heads * self.head_dim
        kv = 2 * self.hidden * self.kv_heads * self.head_dim
        o = self.heads * self.head_dim * self.hidden
        mlp = 3 * self.hidden * self.intermediate          # gate, up, down
        return q + kv + o + mlp

    @property
    def total_params(self) -> int:
        lm_head = 0 if self.tied_embeddings else self.embed_params
        return self.embed_params + lm_head + self.layers * self.layer_params

    @classmethod
    def from_hf_config(cls, cfg: dict, name: str = "") -> "ModelSpec":
        heads = cfg["num_attention_heads"]
        return cls(
            name=name or cfg.get("_name_or_path", "model"),
            layers=cfg["num_hidden_layers"],
            hidden=cfg["hidden_size"],
            heads=heads,
            kv_heads=cfg.get("num_key_value_heads", heads),
            head_dim=cfg.get("head_dim") or cfg["hidden_size"] // heads,
            intermediate=cfg["intermediate_size"],
            vocab=cfg["vocab_size"],
            tied_embeddings=bool(cfg.get("tie_word_embeddings", False)),
            dtype_bytes=DTYPE_BYTES.get(str(cfg.get("torch_dtype", "bfloat16")), 2),
            max_position=cfg.get("max_position_embeddings", 32768),
        )

    @classmethod
    def from_path(cls, path: str | Path, name: str = "") -> "ModelSpec":
        """Read config.json from a local model folder (e.g. an HF cache snapshot)."""
        p = Path(path)
        cfg = json.loads((p / "config.json" if p.is_dir() else p).read_text())
        return cls.from_hf_config(cfg, name=name or p.name)


# Architectures used for calibration (values from each model's config.json).
QWEN25_05B = ModelSpec("Qwen2.5-0.5B-Instruct", layers=24, hidden=896, heads=14, kv_heads=2, head_dim=64,
                       intermediate=4864, vocab=151936, tied_embeddings=True, max_position=32768)
QWEN25_7B = ModelSpec("Qwen2.5-7B-Instruct", layers=28, hidden=3584, heads=28, kv_heads=4, head_dim=128,
                     intermediate=18944, vocab=152064, tied_embeddings=False, max_position=32768)
LLAMA31_8B = ModelSpec("Llama-3.1-8B-Instruct", layers=32, hidden=4096, heads=32, kv_heads=8, head_dim=128,
                       intermediate=14336, vocab=128256, tied_embeddings=False, max_position=131072)
LLAMA31_70B = ModelSpec("Llama-3.1-70B-Instruct", layers=80, hidden=8192, heads=64, kv_heads=8, head_dim=128,
                        intermediate=28672, vocab=128256, tied_embeddings=False, max_position=131072)
