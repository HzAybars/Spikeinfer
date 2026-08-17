"""Make ``spiking_qwen2`` a model type the transformers Auto* classes know.

A spikeinfer model directory carries ``"model_type": "spiking_qwen2"``, which is
honest -- the weights are not loadable as a dense Qwen2. But transformers
resolves a tokenizer by first parsing ``config.json``, and an unregistered
model_type makes that fail with an unhelpful ``'dict' object has no attribute
'model_type'``.

Registering the config class (and pointing it at Qwen2's tokenizer, which is
unchanged by the spiking conversion) makes ``AutoTokenizer.from_pretrained`` work
on a model directory like any other. Registration is idempotent and never
raises: a failure here only costs the convenience of Auto* lookup.
"""
from __future__ import annotations

_registered = False


def register_auto_classes() -> bool:
    """Register ``SpikingQwenConfig`` with AutoConfig/AutoTokenizer."""
    global _registered
    if _registered:
        return True
    try:
        from transformers import AutoConfig, AutoTokenizer
        from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer

        from .reference.modeling_spiking_qwen import SpikingQwenConfig

        try:
            fast_cls = __import__(
                "transformers.models.qwen2.tokenization_qwen2_fast",
                fromlist=["Qwen2TokenizerFast"],
            ).Qwen2TokenizerFast
        except Exception:  # pragma: no cover - tokenizers not installed
            fast_cls = None

        AutoConfig.register(SpikingQwenConfig.model_type, SpikingQwenConfig, exist_ok=True)
        AutoTokenizer.register(
            SpikingQwenConfig,
            slow_tokenizer_class=Qwen2Tokenizer,
            fast_tokenizer_class=fast_cls,
            exist_ok=True,
        )
        _registered = True
    except Exception:  # pragma: no cover - depends on the installed transformers
        return False
    return True
