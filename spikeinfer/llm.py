"""Offline batch API.

    from spikeinfer import LLM, SamplingParams

    llm = LLM("./qwen-spiking")
    for out in llm.generate(["Hello", "World"], SamplingParams(max_tokens=32)):
        print(out.text)

Prompts are submitted all at once and the engine batches them continuously, so
throughput does not depend on the caller interleaving anything.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

from .engine.llm_engine import LLMEngine
from .engine.sequence import RequestOutput
from .sampling_params import SamplingParams


class LLM:
    def __init__(self, model: str, **kwargs) -> None:
        self.engine = LLMEngine.from_pretrained(model, **kwargs)

    @property
    def tokenizer(self):
        return self.engine.tokenizer

    def generate(
        self,
        prompts: str | Sequence[str],
        sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
        prompt_token_ids: Sequence[Sequence[int]] | None = None,
        use_tqdm: bool = False,
    ) -> list[RequestOutput]:
        """Run every prompt to completion and return them in input order."""
        if isinstance(prompts, str):
            prompts = [prompts]
        prompts = list(prompts) if prompts is not None else []
        count = len(prompt_token_ids) if prompt_token_ids is not None else len(prompts)

        if sampling_params is None:
            params_list = [SamplingParams()] * count
        elif isinstance(sampling_params, SamplingParams):
            params_list = [sampling_params] * count
        else:
            params_list = list(sampling_params)
            if len(params_list) != count:
                raise ValueError("sampling_params must match the number of prompts")

        order: list[str] = []
        for i in range(count):
            request_id = self.engine.add_request(
                prompt=prompts[i] if i < len(prompts) else None,
                sampling_params=params_list[i],
                prompt_token_ids=list(prompt_token_ids[i]) if prompt_token_ids else None,
            )
            order.append(request_id)

        results: dict[str, RequestOutput] = {}
        progress = _progress(count) if use_tqdm else None
        while self.engine.has_unfinished_requests():
            for output in self.engine.step():
                if output.finished:
                    results[output.request_id] = output
                    if progress:
                        progress.update(1)
        if progress:
            progress.close()
        return [results[r] for r in order if r in results]

    def stream(
        self,
        prompt: str,
        sampling_params: SamplingParams | None = None,
        prompt_token_ids: list[int] | None = None,
    ) -> Iterator[str]:
        """Yield text deltas for a single prompt as they are produced."""
        request_id = self.engine.add_request(
            prompt=prompt,
            sampling_params=sampling_params or SamplingParams(),
            prompt_token_ids=prompt_token_ids,
        )
        while self.engine.has_unfinished_requests():
            for output in self.engine.step():
                if output.request_id != request_id:
                    continue
                delta = output.outputs[0].delta
                if delta:
                    yield delta
                if output.finished:
                    return

    def chat(
        self,
        messages: list[dict] | list[list[dict]],
        sampling_params: SamplingParams | None = None,
        add_generation_prompt: bool = True,
        **kwargs,
    ) -> list[RequestOutput]:
        """Apply the tokenizer's chat template, then :meth:`generate`."""
        conversations = messages if messages and isinstance(messages[0], list) else [messages]
        prompts = [
            self.apply_chat_template(c, add_generation_prompt=add_generation_prompt, **kwargs)
            for c in conversations
        ]
        return self.generate(prompts, sampling_params)

    def apply_chat_template(
        self, messages: Iterable[dict], add_generation_prompt: bool = True, **kwargs
    ) -> str:
        tokenizer = self.engine.tokenizer
        if tokenizer is None or getattr(tokenizer, "chat_template", None) is None:
            raise RuntimeError(
                "this model has no chat template; call generate() with a raw prompt"
            )
        return tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=add_generation_prompt, **kwargs
        )

    def describe(self) -> dict:
        return self.engine.describe()


def _progress(total: int):  # pragma: no cover - cosmetic
    try:
        from tqdm import tqdm

        return tqdm(total=total, unit="req")
    except ImportError:
        return None
