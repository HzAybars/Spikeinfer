"""Reference (unoptimized) SpikingQwen implementation.

Vendored from the SpikingQwen conversion project so this repository is
self-contained: it is the golden reference the engine is validated against, and
the test suite compares against it directly. It uses snntorch's ``Leaky`` neuron
and the plain ``for t: for layer:`` unroll -- deliberately unoptimized and left
untouched, because its whole job is to be the thing that is known to be correct.

Do not optimize anything in here. Optimize in ``spikeinfer/`` and prove
equivalence against this.
"""
from .lif import LIFNeuron
from .modeling_spiking_qwen import (
    SpikingQwenConfig,
    SpikingQwenForCausalLM,
    SpikingQwenModel,
)

__all__ = [
    "SpikingQwenConfig",
    "SpikingQwenForCausalLM",
    "SpikingQwenModel",
    "LIFNeuron",
]
