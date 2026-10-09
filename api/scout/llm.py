"""Thin wrapper around the OpenAI-compatible LLM API.

One primitive: `call_tool` forces the model to answer through a single tool whose arguments are
a Pydantic model. The output is validated (schema + optional custom check); on failure the error
is fed back to the model and it gets another attempt. Nothing unvalidated leaves this module.
Token usage is accumulated so every caller can report cost.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from scout.config import settings


class LLMError(Exception):
    """The model failed to produce valid output within the allowed attempts."""


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0  # includes reasoning tokens
    reasoning_tokens: int = 0

    def record(self, usage: Any) -> None:
        self.calls += 1
        self.input_tokens += usage.prompt_tokens
        self.output_tokens += usage.completion_tokens
        details = usage.completion_tokens_details
        self.reasoning_tokens += (details.reasoning_tokens or 0) if details else 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            calls=self.calls + other.calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )

    @property
    def cost_usd(self) -> float:
        return (
            self.input_tokens * settings.llm_price_input_per_m
            + self.output_tokens * settings.llm_price_output_per_m
        ) / 1_000_000


_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            max_retries=3,  # network errors and 429s, handled by the SDK
            timeout=180,
        )
    return _client


def call_tool[T: BaseModel](
    messages: list[dict],
    output: type[T],
    *,
    tool_name: str,
    tool_description: str,
    usage: Usage,
    check: Callable[[T], str | None] | None = None,
    max_attempts: int = 3,
    max_tokens: int = 4000,
) -> T:
    """Force a tool call, validate its arguments as `output`, retry with the error on failure.

    `check` runs after schema validation and returns an error message, or None if the result is ok.
    """
    tool = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": tool_description,
            "parameters": output.model_json_schema(),
        },
    }
    messages = list(messages)
    error = "no attempt made"

    for _ in range(max_attempts):
        response = client().chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": tool_name}},
            max_tokens=max_tokens,
            extra_body={"reasoning_split": True},  # MiniMax: keep <think> out of the content
        )
        usage.record(response.usage)
        message = response.choices[0].message

        if not message.tool_calls:
            error = f"you did not call the {tool_name} tool"
            messages.append({"role": "assistant", "content": message.content or ""})
            messages.append({"role": "user", "content": f"Error: {error}. Call it now."})
            continue

        call = message.tool_calls[0]
        try:
            result = output.model_validate_json(call.function.arguments)
            error = check(result) if check else None
        except ValidationError as e:
            error = str(e)
        if error is None:
            return result

        # Show the model its own invalid call plus the error, and let it correct itself.
        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": tool_name, "arguments": call.function.arguments},
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": f"Invalid: {error}\nCall {tool_name} again with corrected arguments.",
            }
        )

    raise LLMError(
        f"{tool_name}: no valid output after {max_attempts} attempts. Last error: {error}"
    )
