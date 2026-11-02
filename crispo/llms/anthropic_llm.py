import os
import threading
from typing import Optional

import anthropic

from crispo.llms import LargeLanguageModel, TYPE_PROMPT


class AnthropicLLM(LargeLanguageModel):
    """
    A Claude model behind the Anthropic Messages API (key from ``ANTHROPIC_API_KEY``, and the workspace from
    ``ANTHROPIC_WORKSPACE_ID`` when the key is not scoped to one). Plain-string prompts are
    sent as a single user message. Only the final answer (the text blocks) is returned; thinking is kept out of it.

    Claude has no temperature setting on the current API, so every role runs at the model's default sampling;
    ``effort`` (low/medium/high/...) sets how much the model thinks before answering.

    Every call is appended to ``self.records`` with the same fields as ``OpenAICompatLLM`` so callers can tally
    cost per role. ``completion_tokens`` includes thinking; ``content_tokens`` is the answer only.
    """

    def __init__(
        self,
        model: str,
        effort: Optional[str] = None,
        max_new_tokens: int = 16000,
        concurrency: int = 16,
        max_retries: int = 10,
    ) -> None:
        super().__init__(max_new_tokens=max_new_tokens, concurrency=concurrency)
        self.model = model
        self.effort = effort
        # a key that is not scoped to a workspace must name one on every request (ANTHROPIC_WORKSPACE_ID)
        workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
        headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
        self.client = anthropic.Anthropic(max_retries=max_retries, timeout=600, default_headers=headers)
        self.records = []
        self._records_lock = threading.Lock()

    def generate(self, prompt: TYPE_PROMPT) -> str:
        messages = (
            [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        )
        extra = {"output_config": {"effort": self.effort}} if self.effort else {}
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_new_tokens,
                messages=messages,
                **extra,
            )
        except (anthropic.InternalServerError, anthropic.OverloadedError, anthropic.ServiceUnavailableError):
            # still failing after the client's retries: count it as an empty output instead of aborting the run,
            # as OpenAICompatLLM does
            with self._records_lock:
                self.records.append(
                    {
                        "head": messages[-1]["content"][:80],
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "content_tokens": 0,
                        "finish_reason": "server_error",
                    }
                )
            return ""
        content = "".join(block.text for block in response.content if block.type == "text")
        usage = response.usage
        thinking_tokens = usage.output_tokens_details.thinking_tokens if usage.output_tokens_details else 0
        with self._records_lock:
            self.records.append(
                {
                    "head": messages[-1]["content"][:80],
                    "prompt_tokens": usage.input_tokens,
                    "completion_tokens": usage.output_tokens,
                    "content_tokens": usage.output_tokens - thinking_tokens,
                    # "length" is what tally() counts as truncated; "refusal" leaves the output empty
                    "finish_reason": "length" if response.stop_reason == "max_tokens" else response.stop_reason,
                }
            )
        return content
