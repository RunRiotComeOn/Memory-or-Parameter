from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


@dataclass
class ModelReply:
    content: str
    reasoning: str | None
    parsed: dict[str, Any]
    usage: dict[str, int | None]
    finish_reason: str | None = None


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        value = json.loads(fenced.group(1))
        if isinstance(value, dict):
            return value

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        value = json.loads(text[start : end + 1])
        if isinstance(value, dict):
            return value
    raise ValueError(f"Model did not return a JSON object: {text[:500]!r}")


class ModelClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
        enable_thinking: bool = False,
        timeout: float = 600.0,
    ) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.seed = seed
        self.enable_thinking = enable_thinking

    def json_chat_messages(self, messages: list[dict[str, str]]) -> ModelReply:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "response_format": {"type": "json_object"},
            "extra_body": {
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
            },
        }
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                # Keep the first request exactly reproducible, but do not replay
                # the same malformed JSON deterministically on recovery attempts.
                kwargs["seed"] = self.seed + attempt
                response = self.client.chat.completions.create(**kwargs)
                message = response.choices[0].message
                content = message.content or ""
                reasoning = getattr(message, "reasoning_content", None)
                usage = response.usage
                return ModelReply(
                    content=content,
                    reasoning=reasoning,
                    parsed=_extract_json(content),
                    usage={
                        "prompt_tokens": getattr(usage, "prompt_tokens", None),
                        "completion_tokens": getattr(usage, "completion_tokens", None),
                        "total_tokens": getattr(usage, "total_tokens", None),
                    },
                )
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(2**attempt)
        assert last_error is not None
        raise last_error

    def chat_messages(self, messages: list[dict[str, str]]) -> ModelReply:
        """Free-form completion. Used by code-acting agents, which emit code, not JSON."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "extra_body": {
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
            },
        }
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                kwargs["seed"] = self.seed + attempt
                response = self.client.chat.completions.create(**kwargs)
                message = response.choices[0].message
                usage = response.usage
                return ModelReply(
                    content=message.content or "",
                    reasoning=getattr(message, "reasoning_content", None),
                    parsed={},
                    finish_reason=getattr(response.choices[0], "finish_reason", None),
                    usage={
                        "prompt_tokens": getattr(usage, "prompt_tokens", None),
                        "completion_tokens": getattr(usage, "completion_tokens", None),
                        "total_tokens": getattr(usage, "total_tokens", None),
                    },
                )
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(2**attempt)
        assert last_error is not None
        raise last_error

    def json_chat(self, *, system: str, user: str) -> ModelReply:
        return self.json_chat_messages(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )
