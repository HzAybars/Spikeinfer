"""Model directory round-trips.

The format is the product's stable surface: someone will publish a converted
model and someone else will load it months later. So the tests check that a
saved directory reloads to exactly the same weights, that tied embeddings
survive the trip (they are stored once but must come back shared), and that a
truncated or foreign checkpoint fails loudly instead of loading a model that is
quietly wrong.
"""
from __future__ import annotations

import json

import pytest
import torch

from conftest import CUDA_AVAILABLE, build_fast_model, small_config
from spikeinfer.loader import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    copy_tokenizer,
    load_config,
    load_model,
    save_config,
    save_model,
)


@pytest.fixture
def saved_model(tmp_path):
    cfg = small_config()
    model = build_fast_model(cfg, "cpu", seed=3)
    save_model(model, cfg, tmp_path)
    return tmp_path, model, cfg


def test_save_writes_the_expected_files(saved_model):
    path, _, _ = saved_model
    assert (path / CONFIG_NAME).exists()
    assert (path / WEIGHTS_NAME).exists()


def test_config_round_trips(saved_model):
    path, _, cfg = saved_model
    loaded = load_config(path)
    for field in ("T", "beta_init", "threshold_init", "hidden_size", "num_hidden_layers",
                  "num_key_value_heads", "vocab_size", "tie_word_embeddings"):
        assert getattr(loaded, field) == getattr(cfg, field), f"{field} changed on round trip"


def test_config_records_the_architecture(saved_model):
    path, _, _ = saved_model
    raw = json.loads((path / CONFIG_NAME).read_text(encoding="utf-8"))
    assert raw["architectures"] == ["FastSpikingQwenForCausalLM"]
    assert raw["model_type"] == "spiking_qwen2"


def test_config_omits_transformers_version(saved_model):
    """Its presence trips a bug in transformers 4.57.2's AutoTokenizer."""
    path, _, _ = saved_model
    raw = json.loads((path / CONFIG_NAME).read_text(encoding="utf-8"))
    assert "transformers_version" not in raw


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA GPU required")
def test_weights_round_trip_exactly(saved_model):
    path, original, _ = saved_model
    loaded, _ = load_model(path, dtype=torch.float32, device="cuda")

    original_state = original.state_dict()
    for name, tensor in loaded.state_dict().items():
        assert torch.equal(tensor.cpu(), original_state[name].cpu()), f"{name} changed"


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA GPU required")
def test_tied_embeddings_stay_shared(saved_model):
    """lm_head is not stored; loading must re-tie it, not leave it random."""
    path, _, cfg = saved_model
    assert cfg.tie_word_embeddings
    loaded, _ = load_model(path, dtype=torch.float32, device="cuda")
    assert loaded.lm_head.weight.data_ptr() == loaded.model.embed_tokens.weight.data_ptr()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA GPU required")
def test_dtype_is_applied_on_load(saved_model):
    path, _, _ = saved_model
    loaded, _ = load_model(path, dtype=torch.bfloat16, device="cuda")
    assert loaded.model.embed_tokens.weight.dtype == torch.bfloat16


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA GPU required")
def test_timesteps_can_be_overridden_at_load(saved_model):
    path, _, _ = saved_model
    _, cfg = load_model(path, dtype=torch.float32, device="cuda", timesteps=2)
    assert cfg.T == 2


def test_loading_a_file_explains_how_to_convert(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"not a directory")
    with pytest.raises(ValueError, match="spikeinfer convert"):
        load_model(checkpoint)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA GPU required")
def test_a_missing_tensor_is_an_error_not_a_random_weight(saved_model, tmp_path):
    from safetensors.torch import load_file, save_file

    path, _, cfg = saved_model
    state = load_file(str(path / WEIGHTS_NAME))
    state.pop("model.layers.0.self_attn.q_proj.weight")

    broken = tmp_path / "broken"
    broken.mkdir()
    save_config(cfg, broken)
    save_file(state, str(broken / WEIGHTS_NAME), metadata={"format": "pt"})

    with pytest.raises(RuntimeError, match="missing"):
        load_model(broken, dtype=torch.float32, device="cuda")


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA GPU required")
def test_an_unknown_tensor_is_an_error(saved_model, tmp_path):
    from safetensors.torch import load_file, save_file

    path, _, cfg = saved_model
    state = load_file(str(path / WEIGHTS_NAME))
    state["model.layers.0.self_attn.not_a_real_weight"] = torch.zeros(4)

    broken = tmp_path / "extra"
    broken.mkdir()
    save_config(cfg, broken)
    save_file(state, str(broken / WEIGHTS_NAME), metadata={"format": "pt"})

    with pytest.raises(RuntimeError, match="unknown"):
        load_model(broken, dtype=torch.float32, device="cuda")


def test_copy_tokenizer_skips_absent_files(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "tokenizer.json").write_text("{}", encoding="utf-8")

    copied = copy_tokenizer(src, dst)
    assert copied == ["tokenizer.json"]
    assert (dst / "tokenizer.json").exists()


def test_copy_tokenizer_on_an_empty_source_is_not_an_error(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    assert copy_tokenizer(src, dst) == []
