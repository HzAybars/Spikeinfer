"""Equivalence tests: FastSpikingQwenForCausalLM vs the reference
SpikingQwenForCausalLM.

The fast (T-batched) path is not bit-exact vs the reference in general.
Folding T into the leading dimension of every GEMM/reduction changes which
cuBLAS/reduction kernel gets selected relative to the reference's T-looped,
per-step execution -- in more than one place: the projection ``nn.Linear``
calls, the batched attention matmul over T*B, and ``Qwen2RMSNorm``'s
``mean(-1)`` reduction. There is no strict/exact mode for T > 1: doing that
properly would mean threading LIF membrane state through a per-timestep
loop, i.e. reimplementing the reference.

Two very different kinds of divergence show up, so this file checks two very
different things:

* Typical relative logit error is tiny (~1e-7), consistent with float
  reassociation noise from kernel-selection differences.
* Occasionally a spike whose membrane lands exactly on its threshold flips
  from a handful of ULPs of upstream noise. That is a discrete event, not
  float noise -- it produces one large, localized jump in an otherwise-tiny
  diff (observed up to ~6.3e-4 relative on one random input at B=2,S=32,T=8;
  reproduced here too, e.g. B=2,S=12,T=4 shows a ~2e-2 absolute jump at a
  single sequence position while every other position differs by ~1e-5).
  This is intrinsic to spiking models -- injecting ~5e-7 relative noise into
  a LIF input flips roughly 1 spike in 2.5 million -- not a bug. A tight
  per-element or per-tensor error bound would therefore flake depending on
  the random input, which is the worst kind of test, so the bound asserted
  below is deliberately loose (1e-2). The invariant that actually holds
  everywhere tested is argmax agreement.

T=1 is the one case that IS bit-exact: with a single timestep there is no T
dimension to batch, so fast and reference run the identical computation.
That is asserted with `torch.equal` as a sharper check alongside the loose
bound.

The sweep covers awkward, non-power-of-two sequence lengths (this is what
originally exposed the localized spike-flip divergence) and is exercised
with a padded ``attention_mask`` (right-padded so the causal mask never
zeroes out every key for a given query, which would otherwise produce NaN
rows in softmax).
"""
from __future__ import annotations

import pytest
import torch

from conftest import build_fast_from_ref, build_ref_model, small_config

S_VALUES = [1, 4, 7, 12, 15, 23, 32]
B_VALUES = [1, 2, 3]
T_VALUES = [1, 2, 4, 8]


def _random_ids(B, S, vocab_size, device, seed):
    torch.manual_seed(seed)
    return torch.randint(0, vocab_size, (B, S), device=device)


def _padded_mask(B, S, device, seed):
    """Right-padded attention_mask: a random valid prefix length per row,
    padding (zeros) on the tail. Never masks position 0, so every causal
    query has at least one valid key -- avoids all-(-inf) softmax rows."""
    mask = torch.ones(B, S, dtype=torch.long, device=device)
    if S > 1:
        torch.manual_seed(seed)
        for b in range(B):
            valid_len = torch.randint(1, S + 1, (1,)).item()
            if valid_len < S:
                mask[b, valid_len:] = 0
    return mask


def _cleanup(*modules):
    for m in modules:
        del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.mark.parametrize("T", T_VALUES)
@pytest.mark.parametrize("B", B_VALUES)
@pytest.mark.parametrize("S", S_VALUES)
def test_fast_close_to_reference(S, B, T, device):
    cfg = small_config(T=T)
    ref = build_ref_model(cfg, device, seed=0)
    fast = build_fast_from_ref(cfg, ref, device)

    ids = _random_ids(B, S, cfg.vocab_size, device, seed=1)
    mask = _padded_mask(B, S, device, seed=2)

    with torch.no_grad():
        ref_logits = ref(ids, attention_mask=mask, T=T)["logits"]
        fast_logits = fast(ids, attention_mask=mask, T=T)["logits"]

    diff = fast_logits - ref_logits
    max_abs_diff = diff.abs().max().item()

    if T == 1:
        # No T dimension to batch -- fast and reference run the identical
        # single-timestep computation, so this must be bit-exact.
        n_diff = (fast_logits != ref_logits).sum().item()
        assert torch.equal(fast_logits, ref_logits), (
            f"T=1 must be bit-exact (no T-batching) for B={B},S={S}: "
            f"{n_diff}/{ref_logits.numel()} elements differ, max abs diff {max_abs_diff:.3e}"
        )

    rel_err = (diff.norm() / ref_logits.norm()).item()
    # Deliberately loose: tolerates an isolated discrete spike-flip jump
    # (see module docstring) without masking a genuine regression.
    assert rel_err < 1e-2, (
        f"relative error too high for B={B},S={S},T={T}: "
        f"rel_err={rel_err:.3e}, max abs diff={max_abs_diff:.3e}"
    )

    ref_argmax = ref_logits.argmax(dim=-1)
    fast_argmax = fast_logits.argmax(dim=-1)
    n_mismatch = (ref_argmax != fast_argmax).sum().item()
    total = ref_argmax.numel()
    assert n_mismatch == 0, (
        f"argmax disagreement for B={B},S={S},T={T}: {n_mismatch}/{total} positions disagree "
        f"({100 * (total - n_mismatch) / total:.2f}% agreement)"
    )

    _cleanup(ref, fast)
