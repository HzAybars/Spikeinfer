"""Producing a spikeinfer model directory.

Two sources:

* **A dense Hugging Face Qwen2 model.** Every real-valued weight transplants
  1:1 -- the spiking architecture keeps Qwen2's linear layers, RMSNorms and RoPE
  untouched and only replaces the pointwise nonlinearities (the SiLU gate, and
  the identity path into q/k/v) with LIF neurons. Only ``beta`` and
  ``threshold`` are new, and they start uniform.

  A transplanted model runs, but its thresholds have not been fitted to the
  activation scales they now gate, so quality is poor until calibration. That
  step is training, and training lives in the conversion project, not in an
  inference engine -- see ``docs/converting.md``.

* **A legacy ``.pt`` checkpoint** from that project, which is already
  calibrated. This is a format change only: pickle to safetensors.
"""
from __future__ import annotations

from pathlib import Path

import torch

from .loader import copy_tokenizer, load_legacy_checkpoint, save_model
from .modeling_fast import FastSpikingQwenForCausalLM
from .reference.modeling_spiking_qwen import SpikingQwenConfig

_SKIP_SUFFIXES = (".lif.beta", ".lif.threshold", ".lif.graded_spikes_factor")


def build_spiking_config(
    src: str | Path, timesteps: int, beta_init: float, threshold_init: float
) -> SpikingQwenConfig:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Config

    base = Qwen2Config.from_pretrained(str(src))
    data = base.to_dict()
    for key in ("architectures", "_name_or_path", "transformers_version", "model_type"):
        data.pop(key, None)
    return SpikingQwenConfig(
        T=timesteps, beta_init=beta_init, threshold_init=threshold_init, **data
    )


def transplant_from_hf(
    src: str | Path,
    out: str | Path,
    timesteps: int = 4,
    beta_init: float = 0.9,
    threshold_init: float = 1.0,
    dtype: torch.dtype = torch.float32,
    verbose: bool = True,
) -> tuple[Path, dict]:
    """Convert a dense Qwen2 checkpoint into an (uncalibrated) spiking one."""
    from transformers import AutoModelForCausalLM

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    config = build_spiking_config(src, timesteps, beta_init, threshold_init)
    dense = AutoModelForCausalLM.from_pretrained(str(src), dtype=dtype)
    spiking = FastSpikingQwenForCausalLM(config).to(dtype)

    dense_state = dense.state_dict()
    target = spiking.state_dict()
    copied, skipped, mismatched = [], [], []
    for name, tensor in target.items():
        if name.endswith(_SKIP_SUFFIXES):
            skipped.append(name)
            continue
        source = dense_state.get(name)
        if source is None:
            skipped.append(name)
        elif source.shape != tensor.shape:
            mismatched.append((name, tuple(source.shape), tuple(tensor.shape)))
        else:
            tensor.copy_(source)
            copied.append(name)

    if mismatched:
        raise RuntimeError(f"shape mismatch on {len(mismatched)} tensors: {mismatched[:3]}")

    spiking.load_state_dict(target)
    save_model(spiking, config, out)
    tok = copy_tokenizer(src, out)

    stats = {
        "copied": len(copied),
        "lif_initialized": len(skipped),
        "tokenizer_files": tok,
        "timesteps": timesteps,
    }
    if verbose:
        print(f"transplanted {len(copied)} tensors, initialised {len(skipped)} LIF parameters")
        print(f"tokenizer files copied: {', '.join(tok) if tok else 'none found'}")
        print(f"wrote {out}")
        print(
            "\nNOTE: thresholds are uniform and uncalibrated. Expect poor quality "
            "until they are fitted -- see docs/converting.md."
        )
    return out, stats


def convert_legacy_checkpoint(
    checkpoint: str | Path,
    out: str | Path,
    tokenizer_src: str | Path | None = None,
    dtype: torch.dtype | None = None,
    verbose: bool = True,
) -> tuple[Path, dict]:
    """Re-serialise a pickled SpikingQwen checkpoint as a model directory."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    state, config = load_legacy_checkpoint(checkpoint)
    if dtype is not None:
        state = {k: v.to(dtype) for k, v in state.items()}

    model = FastSpikingQwenForCausalLM(config)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"checkpoint is missing {len(missing)} tensors, e.g. {missing[:5]}")

    save_model(model, config, out)
    tok = copy_tokenizer(tokenizer_src, out) if tokenizer_src else []

    stats = {
        "tensors": len(state),
        "unexpected": len(unexpected),
        "tokenizer_files": tok,
        "timesteps": config.T,
    }
    if verbose:
        print(f"converted {len(state)} tensors from {checkpoint}")
        if unexpected:
            print(f"  ignored {len(unexpected)} tensors not present in the model")
        print(f"tokenizer files copied: {', '.join(tok) if tok else 'none'}")
        print(f"wrote {out}")
    return out, stats
