"""Core message models for the MEL protocol.

This module provides light-weight dataclasses that mirror the MEL schema.
They provide ergonomic helpers for building requests/results in Python while
remaining compatible with the JSON serialisation used by the wire protocol.

The goal of these helpers is twofold:

* keep the objects easy to debug/inspect (plain dataclasses with repr)
* ensure every request/result can still be validated with the published schema

The classes intentionally avoid pulling in heavy dependencies (e.g. pydantic)
so that they remain usable in constrained on-device environments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from .mel_validate import validate_obj


def _strip_none(data: MutableMapping[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``data`` without keys whose value is ``None``."""

    return {k: v for k, v in data.items() if v is not None}


@dataclass(slots=True)
class TaskInput:
    """Single input item within a task request."""

    name: str
    kind: str
    value: Any | None = None
    lang: Optional[str] = None
    mime: Optional[str] = None
    handle: Optional[str] = None
    size: Optional[Sequence[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "value": self.value,
            "lang": self.lang,
            "mime": self.mime,
            "handle": self.handle,
            "size": list(self.size) if self.size is not None else None,
        }
        return _strip_none(data)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TaskInput":
        return cls(
            name=str(raw.get("name", "")),
            kind=str(raw.get("kind", "")),
            value=raw.get("value"),
            lang=raw.get("lang"),
            mime=raw.get("mime"),
            handle=raw.get("handle"),
            size=tuple(raw["size"]) if raw.get("size") is not None else None,
        )


@dataclass(slots=True)
class TaskExpected:
    """Expected output descriptor for a task."""

    name: str
    kind: str
    schema: Optional[str] = None
    mime: Optional[str] = None
    size: Optional[Sequence[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "name": self.name,
            "kind": self.kind,
            "schema": self.schema,
            "mime": self.mime,
            "size": list(self.size) if self.size is not None else None,
        }
        return _strip_none(data)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TaskExpected":
        return cls(
            name=str(raw.get("name", "")),
            kind=str(raw.get("kind", "")),
            schema=raw.get("schema"),
            mime=raw.get("mime"),
            size=tuple(raw["size"]) if raw.get("size") is not None else None,
        )


_KNOWN_CONSTRAINT_KEYS = {
    "max_latency_ms",
    "max_mem_mb",
    "max_tokens",
    "deterministic",
    "device_pref",
    "privacy",
    "redaction",
}


@dataclass(slots=True)
class TaskConstraints:
    """Container for execution constraints with passthrough for vendor keys."""

    max_latency_ms: Optional[int] = None
    max_mem_mb: Optional[int] = None
    max_tokens: Optional[int] = None
    deterministic: Optional[bool] = None
    device_pref: Optional[Sequence[str]] = None
    privacy: Optional[str] = None
    redaction: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        base = {
            "max_latency_ms": self.max_latency_ms,
            "max_mem_mb": self.max_mem_mb,
            "max_tokens": self.max_tokens,
            "deterministic": self.deterministic,
            "device_pref": list(self.device_pref) if self.device_pref is not None else None,
            "privacy": self.privacy,
            "redaction": self.redaction,
        }
        data = _strip_none(base)
        if self.extras:
            data.update(self.extras)
        return data

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TaskConstraints":
        extras = {k: v for k, v in raw.items() if k not in _KNOWN_CONSTRAINT_KEYS}
        return cls(
            max_latency_ms=raw.get("max_latency_ms"),
            max_mem_mb=raw.get("max_mem_mb"),
            max_tokens=raw.get("max_tokens"),
            deterministic=raw.get("deterministic"),
            device_pref=tuple(raw["device_pref"]) if raw.get("device_pref") is not None else None,
            privacy=raw.get("privacy"),
            redaction=raw.get("redaction"),
            extras=extras,
        )


@dataclass(slots=True)
class Task:
    """High level task description."""

    task_id: str
    intent: str
    inputs: List[TaskInput]
    expected: List[TaskExpected]
    constraints: TaskConstraints
    hints: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "intent": self.intent,
            "inputs": [inp.to_dict() for inp in self.inputs],
            "expected": [exp.to_dict() for exp in self.expected],
            "constraints": self.constraints.to_dict(),
            "hints": dict(self.hints),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Task":
        inputs = [TaskInput.from_dict(item) for item in raw.get("inputs", [])]
        expected = [TaskExpected.from_dict(item) for item in raw.get("expected", [])]
        constraints = TaskConstraints.from_dict(raw.get("constraints", {}))
        hints = dict(raw.get("hints", {}))
        return cls(
            task_id=str(raw.get("task_id", "")),
            intent=str(raw.get("intent", "")),
            inputs=inputs,
            expected=expected,
            constraints=constraints,
            hints=hints,
        )

    def primary_input_text(self) -> Optional[str]:
        """Return the first textual input value, if present."""

        for item in self.inputs:
            if item.kind == "text" and isinstance(item.value, str):
                return item.value
        return None


@dataclass(slots=True)
class TaskOutput:
    name: str
    kind: str
    value: Any | None = None
    handle: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "name": self.name,
            "kind": self.kind,
            "value": self.value,
            "handle": self.handle,
        }
        return _strip_none(data)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TaskOutput":
        return cls(
            name=str(raw.get("name", "")),
            kind=str(raw.get("kind", "")),
            value=raw.get("value"),
            handle=raw.get("handle"),
        )


@dataclass(slots=True)
class TaskError:
    code: str
    msg: str

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "msg": self.msg}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TaskError":
        return cls(code=str(raw.get("code", "")), msg=str(raw.get("msg", "")))


@dataclass(slots=True)
class TaskResult:
    task_id: str
    status: str
    outputs: List[TaskOutput] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[TaskError] = None
    type: str = field(default="TASK_RESULT", init=False)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "type": self.type,
            "task_id": self.task_id,
            "status": self.status,
            "outputs": [out.to_dict() for out in self.outputs],
            "metrics": dict(self.metrics),
            "error": self.error.to_dict() if self.error else None,
        }
        return _strip_none(data)

    def ok(self) -> bool:
        return self.status == "ok"

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], validate: bool = True) -> "TaskResult":
        if validate and not validate_obj(dict(raw)):
            raise ValueError("Payload does not satisfy MEL schema")
        outputs = [TaskOutput.from_dict(item) for item in raw.get("outputs", [])]
        error = TaskError.from_dict(raw["error"]) if raw.get("error") else None
        return cls(
            task_id=str(raw.get("task_id", "")),
            status=str(raw.get("status", "")),
            outputs=outputs,
            metrics=dict(raw.get("metrics", {})),
            error=error,
        )

    @classmethod
    def ok_result(
        cls,
        task_id: str,
        outputs: Iterable[TaskOutput],
        metrics: Optional[Mapping[str, Any]] = None,
    ) -> "TaskResult":
        return cls(task_id=task_id, status="ok", outputs=list(outputs), metrics=dict(metrics or {}))

    @classmethod
    def error_result(
        cls,
        task_id: str,
        code: str,
        msg: str,
        *,
        status: str = "error",
        metrics: Optional[Mapping[str, Any]] = None,
    ) -> "TaskResult":
        return cls(
            task_id=task_id,
            status=status,
            outputs=[],
            metrics=dict(metrics or {}),
            error=TaskError(code=code, msg=msg),
        )


@dataclass(slots=True)
class TaskRequest:
    task: Task
    type: str = field(default="TASK_REQUEST", init=False)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "task": self.task.to_dict()}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], validate: bool = True) -> "TaskRequest":
        if validate and not validate_obj(dict(raw)):
            raise ValueError("Payload does not satisfy MEL schema")
        if raw.get("type") != "TASK_REQUEST":
            raise ValueError(f"Unsupported payload type: {raw.get('type')}")
        task = Task.from_dict(raw.get("task", {}))
        return cls(task=task)


__all__ = [
    "Task",
    "TaskConstraints",
    "TaskError",
    "TaskExpected",
    "TaskInput",
    "TaskOutput",
    "TaskRequest",
    "TaskResult",
]

