"""Bridging helpers for moving between natural language and MEL objects."""

from __future__ import annotations

import uuid
from typing import Any, Dict, Mapping

from .messages import Task, TaskConstraints, TaskExpected, TaskInput, TaskRequest, TaskResult


INTENTS = {"qa", "chat", "classify", "summarize", "detect", "generate.image", "generate.audio"}


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def english_to_mel(
    text: str,
    intent: str = "qa",
    *,
    max_latency_ms: int = 200,
    max_tokens: int = 128,
    device_pref: tuple[str, ...] = ("NPU", "CPU"),
    deterministic: bool = True,
    return_dict: bool = True,
) -> TaskRequest | Dict[str, Any]:
    """Create a MEL :class:`TaskRequest` from a natural-language prompt."""

    if intent not in INTENTS:
        raise ValueError(f"Unsupported intent: {intent}")
    request = TaskRequest(
        task=Task(
            task_id=_new_id("T"),
            intent=intent,
            inputs=[
                TaskInput(name="text", kind="text", lang="en", value=text.strip()),
            ],
            expected=[TaskExpected(name="answer", kind="text")],
            constraints=TaskConstraints(
                max_latency_ms=max_latency_ms,
                max_tokens=max_tokens,
                device_pref=device_pref,
                deterministic=deterministic,
            ),
            hints={},
        )
    )
    return request.to_dict() if return_dict else request


def mel_to_english(result: Mapping[str, Any] | TaskResult) -> str:
    """Extract the textual answer from a MEL result payload."""

    if isinstance(result, TaskResult):
        data = result
    else:
        try:
            data = TaskResult.from_dict(dict(result))
        except Exception:
            return ""
    if data.status != "ok":
        if data.error:
            return f"Error: {data.error.code} {data.error.msg}".strip()
        return ""
    for out in data.outputs:
        if out.name == "answer" and out.kind == "text" and isinstance(out.value, str):
            return out.value.strip()
    return ""


__all__ = ["english_to_mel", "mel_to_english", "INTENTS"]

