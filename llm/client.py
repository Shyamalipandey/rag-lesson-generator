"""LLM provider layer.

Three providers share one tiny interface, `complete(task, system, prompt) -> str`:

* anthropic - Claude through the Anthropic API  (needs ANTHROPIC_API_KEY)
* openai    - OpenAI, or ANY OpenAI-compatible server (set OPENAI_BASE_URL)  (needs OPENAI_API_KEY)
* offline   - no API, no cost. Reference lesson + a rule-based evaluator. For tests and demos only.

`task` is "generate", "evaluate" or "regenerate". Real providers ignore it; the offline one needs it.
"""
import os
import time
from abc import ABC, abstractmethod
from typing import Callable, Optional


class LLMError(RuntimeError):
    """Raised when an LLM call fails for good (after retries) or returns nothing."""


class BaseLLM(ABC):
    name = "base"

    @abstractmethod
    def complete(self, task: str, system: str, prompt: str) -> str:
        """Send one request and return the model's text answer."""


def _with_retries(call: Callable[[], str], retries: int, delay: float, label: str) -> str:
    """Run `call`; on any error wait (1x, 2x, 4x ... delay) and try again, then give up with LLMError."""
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            return call()
        except Exception as exc:  # network errors, rate limits, empty answers ...
            last_error = exc
            if attempt < retries:
                time.sleep(delay * (2 ** attempt))
    raise LLMError(f"{label} call failed after {retries + 1} attempt(s): {last_error}") from last_error


class AnthropicLLM(BaseLLM):
    name = "anthropic"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 max_tokens: int = 8000, retries: int = 2, retry_delay: float = 2.0):
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")
        import anthropic  # imported here so the offline mode works without the package

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
        self.max_tokens = max_tokens
        self.retries = retries
        self.retry_delay = retry_delay

    def complete(self, task: str, system: str, prompt: str) -> str:
        def call() -> str:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
            if not text.strip():
                raise ValueError("The model returned an empty answer.")
            return text

        return _with_retries(call, self.retries, self.retry_delay, "Anthropic")


class OpenAILLM(BaseLLM):
    """OpenAI, or any server that speaks the OpenAI chat API (set OPENAI_BASE_URL)."""

    name = "openai"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 base_url: Optional[str] = None, retries: int = 2, retry_delay: float = 2.0):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMError("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
        import openai

        self.client = openai.OpenAI(api_key=api_key, base_url=base_url or os.getenv("OPENAI_BASE_URL") or None)
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.retries = retries
        self.retry_delay = retry_delay

    def complete(self, task: str, system: str, prompt: str) -> str:
        def call() -> str:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            )
            text = response.choices[0].message.content or ""
            if not text.strip():
                raise ValueError("The model returned an empty answer.")
            return text

        return _with_retries(call, self.retries, self.retry_delay, "OpenAI")


def get_llm(provider: Optional[str] = None) -> BaseLLM:
    """Pick a provider: explicit argument > LLM_PROVIDER env var > whichever API key exists > offline."""
    provider = (provider or os.getenv("LLM_PROVIDER") or "").strip().lower()
    if provider in ("", "auto"):
        if os.getenv("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        else:
            provider = "offline"
    if provider == "anthropic":
        return AnthropicLLM()
    if provider == "openai":
        return OpenAILLM()
    if provider == "offline":
        from llm.offline import OfflineLLM

        return OfflineLLM()
    raise LLMError(f"Unknown LLM provider '{provider}'. Use anthropic, openai or offline.")
