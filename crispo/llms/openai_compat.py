import threading
from typing import Optional

from openai import InternalServerError, OpenAI
from tokenizers import Tokenizer

from crispo.llms import LargeLanguageModel, TYPE_PROMPT


class OpenAICompatLLM(LargeLanguageModel):
    """
    An LLM served behind an OpenAI-compatible chat endpoint (e.g. vLLM). Plain-string prompts are sent as a
    single user message. Only the final answer (``content``) is returned; reasoning is kept out of it.

    Every call is appended to ``self.records`` with its token usage so callers can tally cost per role.
    ``content_tokens`` is counted with ``tokenizer_file``; the rest of ``completion_tokens`` is reasoning
    (vLLM reports both together).
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        tokenizer_file: str,
        reasoning_effort: Optional[str] = None,
        max_new_tokens: int = 8192,
        temperature: float = 0.0,
        concurrency: int = 16,
    ) -> None:
        super().__init__(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            concurrency=concurrency,
        )
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=600)
        self.tokenizer = Tokenizer.from_file(tokenizer_file)
        self.records = []
        self._records_lock = threading.Lock()

    def generate(self, prompt: TYPE_PROMPT) -> str:
        messages = (
            [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        )
        extra = (
            {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
                **extra,
            )
        except InternalServerError:
            # vLLM returns 500 when gpt-oss emits malformed harmony tokens; at temperature 0 the client's
            # retries hit it again. Count it as an empty output instead of aborting the whole run.
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
        choice = response.choices[0]
        content = choice.message.content or ""
        content_tokens = len(self.tokenizer.encode(content).ids)
        with self._records_lock:
            self.records.append(
                {
                    "head": messages[-1]["content"][:80],
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "content_tokens": content_tokens,
                    "finish_reason": choice.finish_reason,
                }
            )
        return content
