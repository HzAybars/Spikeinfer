"""Loading and saving spikeinfer models.

The on-disk format is a directory, deliberately shaped like a Hugging Face one
so the tokenizer files can sit next to the weights and existing tooling can read
it::

    model/
      config.json              SpikingQwenConfig, model_type "spiking_qwen2"
      model.safetensors        weights (single file or sharded + index)
      tokenizer.json, tokenizer_config.json, ...

safetensors rather than ``torch.save``: the pickled ``.pt`` format executes
arbitrary code on load, which is not acceptable for a server that may be pointed
at a downloaded checkpoint. Legacy ``.pt`` checkpoints are still readable
through :func:`load_legacy_checkpoint`, but only explicitly, and
``spikeinfer convert`` turns one into a directory once and for all.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .hf_compat import register_auto_classes
from .kernels import validate_thresholds
from .modeling_fast import FastSpikingQwenForCausalLM
from .reference.modeling_spiking_qwen import SpikingQwenConfig

register_auto_classes()

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"
INDEX_NAME = "model.safetensors.index.json"

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "generation_config.json",
    "chat_template.jinja",
)


def load_config(path: str | Path) -> SpikingQwenConfig:
    path = Path(path)
    config_file = path / CONFIG_NAME if path.is_dir() else path
    with open(config_file, encoding="utf-8") as fh:
        raw = json.load(fh)
    raw.pop("architectures", None)
    raw.pop("_name_or_path", None)
    return SpikingQwenConfig(**raw)


def save_config(config: SpikingQwenConfig, path: str | Path) -> None:
    data = config.to_dict()
    data["architectures"] = ["FastSpikingQwenForCausalLM"]
    data["model_type"] = "spiking_qwen2"
    # transformers 4.57.2 has a bug in AutoTokenizer.from_pretrained: when a
    # local config.json declares transformers_version <= 4.57.2 it does
    # `_config.model_type` on a plain dict and raises AttributeError. The field
    # is provenance metadata we do not read, so omitting it costs nothing and
    # keeps model directories loadable on that release.
    data.pop("transformers_version", None)
    with open(Path(path) / CONFIG_NAME, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)


def _read_state_dict(path: Path, device: str) -> dict[str, torch.Tensor]:
    index = path / INDEX_NAME
    if index.exists():
        with open(index, encoding="utf-8") as fh:
            shards = sorted(set(json.load(fh)["weight_map"].values()))
        state: dict[str, torch.Tensor] = {}
        for shard in shards:
            state.update(load_file(path / shard, device=device))
        return state
    return load_file(path / WEIGHTS_NAME, device=device)


def _build_empty(config: SpikingQwenConfig) -> FastSpikingQwenForCausalLM:
    """Construct without allocating weights, falling back to a real CPU build.

    Meta-device construction avoids materialising a full fp32 copy of the model
    before the real weights arrive. snntorch's ``Leaky`` is third-party and not
    guaranteed to be meta-safe, so there is a fallback.
    """
    try:
        with torch.device("meta"):
            return FastSpikingQwenForCausalLM(config)
    except (NotImplementedError, RuntimeError):  # pragma: no cover - snntorch dependent
        return FastSpikingQwenForCausalLM(config)


def load_model(
    path: str | Path,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cuda",
    timesteps: int | None = None,
) -> tuple[FastSpikingQwenForCausalLM, SpikingQwenConfig]:
    """Load a spikeinfer model directory onto ``device`` in ``dtype``."""
    path = Path(path)
    if path.is_file():
        raise ValueError(
            f"{path} is a file, not a model directory. Convert it first:\n"
            f"    spikeinfer convert --from-checkpoint {path} --out <dir> --tokenizer <hf-model>"
        )

    config = load_config(path)
    if timesteps is not None:
        config.T = timesteps

    model = _build_empty(config)
    state = _read_state_dict(path, device="cpu")
    state = {k: v.to(dtype=dtype) for k, v in state.items()}

    if config.tie_word_embeddings and "lm_head.weight" not in state:
        state["lm_head.weight"] = state["model.embed_tokens.weight"]

    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if missing:
        raise RuntimeError(f"checkpoint is missing {len(missing)} tensors, e.g. {missing[:5]}")
    if unexpected:
        raise RuntimeError(f"checkpoint has {len(unexpected)} unknown tensors, e.g. {unexpected[:5]}")

    model = model.to(device=device)
    if config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    model.eval()
    validate_thresholds(model)
    return model, config


def save_model(
    model: FastSpikingQwenForCausalLM,
    config: SpikingQwenConfig,
    path: str | Path,
) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    save_config(config, path)

    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    if config.tie_word_embeddings:
        state.pop("lm_head.weight", None)
    save_file(state, str(path / WEIGHTS_NAME), metadata={"format": "pt"})


def load_legacy_checkpoint(path: str | Path) -> tuple[dict, SpikingQwenConfig]:
    """Read a ``torch.save({"config", "state_dict"})`` checkpoint.

    Unpickling executes code from the file. Only call this on checkpoints you
    produced yourself -- which is why the engine never calls it and
    ``spikeinfer convert`` asks for it by name.
    """
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if "state_dict" not in ckpt or "config" not in ckpt:
        raise ValueError(f"{path} is not a SpikingQwen checkpoint (needs config + state_dict)")
    return ckpt["state_dict"], SpikingQwenConfig(**ckpt["config"])


def copy_tokenizer(src: str | Path, dst: str | Path) -> list[str]:
    """Copy tokenizer/config sidecar files, skipping ones that do not exist."""
    import shutil

    src, dst = Path(src), Path(dst)
    copied = []
    for name in TOKENIZER_FILES:
        candidate = src / name
        if candidate.exists():
            shutil.copy2(candidate, dst / name)
            copied.append(name)
    return copied
