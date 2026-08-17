# Converting a model

spikeinfer serves spiking models; it does not create them. This page explains what
`spikeinfer convert` does, what it deliberately does not do, and what you have to do
yourself to get a model that produces sensible text.

## The model directory

```
qwen-spiking/
  config.json           SpikingQwenConfig, model_type "spiking_qwen2"
  model.safetensors     weights
  tokenizer.json, tokenizer_config.json, vocab.json, merges.txt, ...
```

safetensors rather than `torch.save`: the pickled `.pt` format executes arbitrary code on
load, which is not acceptable for a server that may be pointed at a downloaded checkpoint.
The engine never unpickles anything. `spikeinfer convert --from-checkpoint` reads a legacy
`.pt` once, explicitly, so you never have to again.

`config.json` deliberately omits `transformers_version`. transformers 4.57.2 has a bug in
`AutoTokenizer.from_pretrained` — it reads a local `config.json` into a dict and then calls
`.model_type` on it — which fires whenever that key is present and declares a version
≤ 4.57.2. The field is provenance metadata nothing here reads, so dropping it costs
nothing and keeps model directories loadable on that release.

## From a dense Qwen2 model

```bash
spikeinfer convert --from-hf Qwen/Qwen2.5-0.5B --out ./qwen-spiking --timesteps 4
```

The spiking architecture keeps Qwen2's linear layers, RMSNorms and RoPE untouched and
replaces only the pointwise nonlinearities — the SiLU gate, and the identity path into
q/k/v — with LIF neurons unrolled over `T` timesteps. So every real-valued weight
transplants 1:1 by name, and the only new parameters are each neuron's per-channel `beta`
(decay) and `threshold`.

Those start uniform: `beta = 0.9`, `threshold = 1.0`.

## Why the result does not work yet

A threshold is a decision boundary against an activation scale. Uniform thresholds are
calibrated against nothing, so most neurons either saturate (fire every timestep) or stay
silent, and the rate code carries almost no information. The model is fluent-looking noise.

Fixing that is **training**, not inference:

1. **Threshold calibration.** Run a small corpus through the model and set each neuron's
   threshold from the distribution of its input current — a percentile of the pre-spike
   activation is the usual starting point. Floor thresholds at something small and
   positive; the kernel requires `threshold > 0` (a non-positive threshold makes the `t=0`
   reset fire, because `H(0 − θ) = 1` when `θ ≤ 0`), and `validate_thresholds()` checks
   this at load time.
2. **Fine-tuning.** Train with a surrogate gradient (snntorch's `fast_sigmoid` is what the
   reference uses) on a causal-LM objective, learning `beta` and `threshold` alongside the
   transplanted weights.

Both run against `spikeinfer/reference/`, the unoptimized snntorch implementation — it has
a backward pass and the engine does not. Parameter names match exactly, so a model trained
on the reference path loads into the engine with no key translation:

```python
from spikeinfer import save_model
from spikeinfer.reference import SpikingQwenForCausalLM

# ... calibrate / fine-tune reference_model ...
save_model(reference_model, config, "./qwen-spiking")
```

## Choosing T

`T` is the number of timesteps the LIF recurrence is unrolled over, and it is the main
quality/cost dial:

- **cost is linear in T** — T× the FLOPs, T× the KV cache, T× the LIF work;
- **quality rises and saturates.** More timesteps means a finer rate code, but each one
  carries less new information than the last.

T=4 is what this project was developed against. You can serve a checkpoint at a lower `T`
than it was trained for with `--timesteps`, which is faster and worse; there is no
mechanism for serving it at a higher one meaningfully, since the neurons were fitted at
the training value.

## Verifying a conversion

```bash
spikeinfer info ./qwen-spiking          # architecture and cache math, no weights loaded
spikeinfer generate ./qwen-spiking -p "The capital of France is" --temperature 0
```

The engine calls `validate_thresholds()` on load and refuses a model with a non-positive
threshold anywhere, which catches the most common calibration mistake immediately.

To confirm the *engine* is faithful to your checkpoint rather than that your checkpoint is
good, compare against the eager path — that is what
`tests/test_engine.py::test_greedy_matches_the_eager_path` does, and it is worth running
on a real checkpoint after any conversion change:

```python
import torch
from spikeinfer import LLM, SamplingParams
from spikeinfer.kv_cache import generate as eager_generate

llm = LLM("./qwen-spiking")
prompt = "The capital of France is"
ids = torch.tensor([llm.tokenizer.encode(prompt)], device="cuda")

engine_ids = llm.generate([prompt], SamplingParams(max_tokens=24, temperature=0.0,
                                                   ignore_eos=True))[0].outputs[0].token_ids
eager_ids = eager_generate(llm.engine.model, ids, max_new_tokens=24,
                           temperature=0.0, use_sdpa=True)[0, ids.shape[1]:].tolist()
assert engine_ids == eager_ids
```

## Other architectures

The kernels are architecture-agnostic — `lif_multistep`, `pack_spikes` and
`paged_spike_attn_decode` know about timesteps, channels and head dimensions, nothing
else. The model class is not: `modeling_fast.py` is Qwen2-shaped (GQA supported).

Porting another decoder means writing a model class with `forward_paged`, following the
pattern in `modeling_fast.py`: project, apply RoPE **before** the neuron (rotating a binary
vector does not give a binary vector), spike, then hand `q`/`k`/`v` to `paged_attention`.
