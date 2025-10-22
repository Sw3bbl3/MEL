"""High level runtime primitives for orchestrating MEL agents.

The runtime is intentionally lightweight so that it can run on-device.  It
supports:

* registering multiple agents with intent support declarations
* scoring and selecting the best agent for a task
* tiny conversation memory (windowed history per session)
* JSON serialisation using :mod:`mel.messages`

It is designed to be embedded inside applications (HTTP servers, CLIs, batch
pipelines) without imposing heavy dependencies.
"""

from __future__ import annotations

import json
import os
import warnings
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from .messages import Task, TaskOutput, TaskRequest, TaskResult


class AgentError(RuntimeError):
    """Generic failure raised by an agent."""


class CannotHandleTask(AgentError):
    """Raised by agents when they elect not to handle a task."""


@dataclass(slots=True)
class TaskContext:
    """Context passed to agents during execution."""

    session_id: Optional[str]
    history: Sequence[Mapping[str, str]] = field(default_factory=tuple)


class Agent:
    """Base class for router agents."""

    def __init__(
        self,
        *,
        name: str,
        intents: Optional[Iterable[str]] = None,
        priority: int = 0,
        description: Optional[str] = None,
    ) -> None:
        self.name = name
        self._intents = tuple(intents or ("*",))
        self.priority = priority
        self.description = description or ""

    # -- selection helpers -------------------------------------------------
    def supports_intent(self, intent: str) -> bool:
        if "*" in self._intents:
            return True
        return intent in self._intents

    def score_task(self, task: Task, ctx: TaskContext) -> float:
        """Return a score for ``task``.

        Higher is better.  The default implementation simply returns the
        ``priority`` if the agent supports the intent, or ``-inf`` otherwise.
        """

        if not self.supports_intent(task.intent):
            return float("-inf")
        return float(self.priority)

    # -- execution ---------------------------------------------------------
    def handle(self, task: Task, ctx: TaskContext) -> TaskResult:
        raise NotImplementedError

    # -- metadata ----------------------------------------------------------
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "intents": list(self._intents),
            "priority": self.priority,
            "description": self.description,
            "agent_type": type(self).__name__,
        }


class RuleAgent(Agent):
    """Simple pattern matching agent with deterministic responses."""

    @dataclass(slots=True)
    class _Rule:
        name: str
        response: str
        contains_any: Sequence[str] = ()
        regex: Optional[str] = None
        intent: Optional[str] = None

    def __init__(self, *, name: str, rules: Sequence[Mapping[str, Any]], **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)
        self._rules = [self._parse_rule(rule) for rule in rules]

    def _parse_rule(self, raw: Mapping[str, Any]) -> "RuleAgent._Rule":
        contains_any = tuple(s.lower() for s in raw.get("match", {}).get("contains_any", []))
        regex = raw.get("match", {}).get("regex")
        return RuleAgent._Rule(
            name=str(raw.get("name", raw.get("id", "rule"))),
            response=str(raw.get("response", "")),
            contains_any=contains_any,
            regex=regex,
            intent=raw.get("intent"),
        )

    def score_task(self, task: Task, ctx: TaskContext) -> float:
        if not self.supports_intent(task.intent):
            return float("-inf")
        return float(self.priority + 10)  # deterministic fast path

    def handle(self, task: Task, ctx: TaskContext) -> TaskResult:
        query = task.primary_input_text() or ""
        q_lower = query.lower()
        for rule in self._rules:
            if rule.intent and rule.intent != task.intent:
                continue
            if rule.contains_any and not any(token in q_lower for token in rule.contains_any):
                continue
            if rule.regex:
                import re

                if not re.search(rule.regex, query, re.IGNORECASE):
                    continue
            outputs = [TaskOutput(name="answer", kind="text", value=rule.response)]
            metrics = {
                "agent": self.name,
                "rule": rule.name,
                "latency_ms": 3,
            }
            return TaskResult.ok_result(task.task_id, outputs, metrics)
        raise CannotHandleTask(f"{self.name} has no rule for task {task.task_id}")


class EchoAgent(Agent):
    """Fallback agent that mirrors the primary input."""

    def handle(self, task: Task, ctx: TaskContext) -> TaskResult:
        text = task.primary_input_text() or ""
        outputs = [TaskOutput(name="answer", kind="text", value=text)]
        metrics = {"agent": self.name, "mode": "echo", "latency_ms": 2}
        return TaskResult.ok_result(task.task_id, outputs, metrics)


class HFTextGenerationAgent(Agent):
    """Lazy text-generation agent powered by Hugging Face pipelines."""

    def __init__(
        self,
        *,
        name: str,
        model_id: str,
        intents: Optional[Iterable[str]] = None,
        priority: int = 20,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_new_tokens: int = 192,
        top_p: float = 0.9,
        description: Optional[str] = None,
    ) -> None:
        super().__init__(name=name, intents=intents, priority=priority, description=description)
        self.model_id = model_id
        self.system_prompt = system_prompt or (
            "You are an on-device assistant. Respond accurately and concisely."
        )
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self._lock = threading.Lock()
        self._pipeline = None

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        with self._lock:
            if self._pipeline is not None:
                return self._pipeline
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            except ImportError as exc:  # pragma: no cover - optional dependency path
                raise AgentError("transformers is required for HFTextGenerationAgent") from exc

            device = 0 if torch.cuda.is_available() else -1
            torch.set_num_threads(max(1, os.cpu_count() or 1))
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            self._pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=device,
            )
            return self._pipeline

    def handle(self, task: Task, ctx: TaskContext) -> TaskResult:
        if not self.supports_intent(task.intent):
            raise CannotHandleTask(f"Intent {task.intent} unsupported")
        prompt = self._build_prompt(task, ctx)
        pipe = self._ensure_pipeline()
        generated = pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            eos_token_id=pipe.tokenizer.eos_token_id,
            pad_token_id=pipe.tokenizer.eos_token_id,
        )[0]["generated_text"]
        answer = self._extract_answer(generated)
        outputs = [TaskOutput(name="answer", kind="text", value=answer)]
        metrics = {"agent": self.name, "mode": "hf", "model_id": self.model_id}
        return TaskResult.ok_result(task.task_id, outputs, metrics)

    def _build_prompt(self, task: Task, ctx: TaskContext) -> str:
        lines = [self.system_prompt.strip(), ""]
        for turn in ctx.history:
            role = turn.get("role")
            text = turn.get("text", "")
            if role == "user":
                lines.append(f"User: {text}")
            elif role == "assistant":
                lines.append(f"Assistant: {text}")
        query = task.primary_input_text() or ""
        lines.append(f"User: {query}")
        lines.append("Assistant:")
        return "\n".join(lines)

    def _extract_answer(self, generated: str) -> str:
        marker = "Assistant:"
        pos = generated.rfind(marker)
        if pos >= 0:
            return generated[pos + len(marker):].strip().split("\n\n")[0].strip()
        return generated.strip().split("\n\n")[0].strip()


class ConversationMemory:
    """Tiny in-memory conversation store keyed by session ID."""

    def __init__(self, *, window: int = 6) -> None:
        self._window = window
        self._entries: Dict[str, List[Mapping[str, str]]] = {}
        self._lock = threading.Lock()

    def get(self, session_id: Optional[str]) -> Sequence[Mapping[str, str]]:
        if not session_id:
            return ()
        with self._lock:
            return tuple(self._entries.get(session_id, ()))

    def append(self, session_id: Optional[str], role: str, text: str) -> None:
        if not session_id:
            return
        with self._lock:
            history = self._entries.setdefault(session_id, [])
            history.append({"role": role, "text": text})
            if len(history) > self._window:
                del history[: len(history) - self._window]


class EventRecorder:
    """Minimal event sink used by the runtime for observability."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: List[Dict[str, Any]] = []

    def record(self, event_type: str, **payload: Any) -> None:
        data = {"type": event_type, "ts": time.time(), **payload}
        with self._lock:
            self._events.append(data)

    def snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._events)


class RouterRuntime:
    """Coordinator that selects and executes agents for MEL tasks."""

    def __init__(
        self,
        *,
        agents: Optional[Iterable[Agent]] = None,
        memory: Optional[ConversationMemory] = None,
        validate: bool = True,
        recorder: Optional[EventRecorder] = None,
    ) -> None:
        self._agents: List[Agent] = sorted(list(agents or []), key=lambda a: a.priority, reverse=True)
        self._lock = threading.Lock()
        self._memory = memory or ConversationMemory()
        self._validate = validate
        self._recorder = recorder or EventRecorder()

    # -- agent management --------------------------------------------------
    def register(self, agent: Agent) -> None:
        with self._lock:
            self._agents.append(agent)
            self._agents.sort(key=lambda a: a.priority, reverse=True)

    @property
    def agents(self) -> Sequence[Agent]:
        with self._lock:
            return tuple(self._agents)

    # -- execution ---------------------------------------------------------
    def handle(self, payload: Mapping[str, Any] | TaskRequest) -> TaskResult:
        request = self._ensure_request(payload)
        task = request.task
        session_id = self._resolve_session(task)
        ctx = TaskContext(session_id=session_id, history=self._memory.get(session_id))
        agent = self._select_agent(task, ctx)
        if agent is None:
            self._recorder.record("router.no_agent", task_id=task.task_id, intent=task.intent)
            return TaskResult.error_result(task.task_id, code="router.no_agent", msg="No agent available")

        start = time.time()
        try:
            result = agent.handle(task, ctx)
        except CannotHandleTask:
            self._recorder.record(
                "agent.cannot_handle",
                agent=agent.name,
                task_id=task.task_id,
                intent=task.intent,
            )
            return TaskResult.error_result(task.task_id, code="agent.cannot_handle", msg=f"{agent.name} rejected task")
        except Exception as exc:  # pragma: no cover - defensive path
            self._recorder.record(
                "agent.error",
                agent=agent.name,
                task_id=task.task_id,
                intent=task.intent,
                error=str(exc),
            )
            return TaskResult.error_result(task.task_id, code="agent.error", msg=str(exc))

        latency_ms = max(1, int((time.time() - start) * 1000))
        result.metrics.setdefault("agent", agent.name)
        result.metrics.setdefault("latency_ms", latency_ms)
        self._update_memory(session_id, task, result)
        self._recorder.record(
            "router.success",
            agent=agent.name,
            task_id=task.task_id,
            intent=task.intent,
            latency_ms=latency_ms,
        )
        return result

    def _ensure_request(self, payload: Mapping[str, Any] | TaskRequest) -> TaskRequest:
        if isinstance(payload, TaskRequest):
            return payload
        return TaskRequest.from_dict(dict(payload), validate=self._validate)

    def _resolve_session(self, task: Task) -> Optional[str]:
        return task.hints.get("session_id") or task.hints.get("conversation_id")

    def _select_agent(self, task: Task, ctx: TaskContext) -> Optional[Agent]:
        candidates: List[tuple[float, Agent]] = []
        for agent in self.agents:
            score = agent.score_task(task, ctx)
            if score == float("-inf"):
                continue
            candidates.append((score, agent))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _update_memory(self, session_id: Optional[str], task: Task, result: TaskResult) -> None:
        if not session_id:
            return
        text = task.primary_input_text()
        if text:
            self._memory.append(session_id, "user", text)
        answer = None
        for output in result.outputs:
            if output.kind == "text" and isinstance(output.value, str):
                answer = output.value
                break
        if answer:
            self._memory.append(session_id, "assistant", answer)

    # -- configuration -----------------------------------------------------
    @classmethod
    def with_defaults(cls) -> "RouterRuntime":
        """Return a runtime with opinionated built-in agents."""

        default_rules = [
            {
                "name": "elbrus",
                "match": {"contains_any": ["tallest", "mountain", "europe"]},
                "response": "Mount Elbrus, 5,642 m",
                "intent": "qa",
            },
            {
                "name": "greeting",
                "match": {"contains_any": ["hello", "hi", "hey"]},
                "response": "Hi! How can I help?",
                "intent": "chat",
            },
        ]
        agents: List[Agent] = [
            RuleAgent(name="rules.primary", rules=default_rules, intents=("qa", "chat"), priority=50),
        ]
        model_id = os.environ.get("MEL_TEXT_MODEL")
        if model_id:
            try:
                agents.append(
                    HFTextGenerationAgent(
                        name="hf.generate",
                        model_id=model_id,
                        intents=("qa", "chat", "summarize"),
                        priority=30,
                    )
                )
            except Exception as exc:  # pragma: no cover - optional dependency path
                warnings.warn(
                    f"Failed to initialise HFTextGenerationAgent: {exc}",
                    RuntimeWarning,
                )
        agents.append(
            EchoAgent(name="echo.fallback", intents=("qa", "chat", "classify", "summarize"), priority=1)
        )
        return cls(agents=agents)

    # -- persistence -------------------------------------------------------
    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "RouterRuntime":
        """Build a runtime from a configuration mapping."""

        memory_cfg = config.get("conversation_memory", {})
        memory = ConversationMemory(window=int(memory_cfg.get("window", 6)))
        agents_cfg = config.get("agents", [])
        agents = [build_agent_from_config(agent_cfg) for agent_cfg in agents_cfg]
        agents = [agent for agent in agents if agent is not None]
        validate = bool(config.get("validate", True))
        return cls(agents=agents, memory=memory, validate=validate)

    @classmethod
    def from_config_file(cls, path: str | Path) -> "RouterRuntime":
        data = load_config_file(path)
        return cls.from_config(data)

    def export_state(self) -> Dict[str, Any]:
        return {
            "agents": [agent.metadata() for agent in self.agents],
            "events": self._recorder.snapshot(),
        }


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

_AGENT_BUILDERS: Dict[str, Callable[[Mapping[str, Any]], Optional[Agent]]] = {}


def register_agent_builder(name: str) -> Callable[[Callable[[Mapping[str, Any]], Optional[Agent]]], Callable[[Mapping[str, Any]], Optional[Agent]]]:
    def decorator(fn: Callable[[Mapping[str, Any]], Optional[Agent]]):
        _AGENT_BUILDERS[name] = fn
        return fn

    return decorator


def build_agent_from_config(cfg: Mapping[str, Any]) -> Optional[Agent]:
    agent_type = cfg.get("type")
    if not agent_type:
        raise ValueError("Agent configuration requires a 'type' field")
    builder = _AGENT_BUILDERS.get(agent_type)
    if builder is None:
        raise ValueError(f"Unknown agent type: {agent_type}")
    agent = builder(cfg)
    return agent


@register_agent_builder("echo")
def _build_echo_agent(cfg: Mapping[str, Any]) -> Agent:
    return EchoAgent(
        name=str(cfg.get("name", "echo")),
        intents=cfg.get("intents"),
        priority=int(cfg.get("priority", 0)),
        description=cfg.get("description"),
    )


@register_agent_builder("rules")
def _build_rule_agent(cfg: Mapping[str, Any]) -> Agent:
    rules = cfg.get("rules")
    if not isinstance(rules, Sequence):
        raise ValueError("Rule agent requires a 'rules' array")
    return RuleAgent(
        name=str(cfg.get("name", "rules")),
        intents=cfg.get("intents"),
        priority=int(cfg.get("priority", 10)),
        description=cfg.get("description"),
        rules=rules,
    )


def load_config_file(path: str | Path) -> Dict[str, Any]:
    """Load a runtime configuration from JSON or TOML."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".json", ""}:
        return json.loads(text)
    if path.suffix.lower() == ".toml":
        import tomllib

        return tomllib.loads(text)
    raise ValueError(f"Unsupported config extension: {path.suffix}")


__all__ = [
    "Agent",
    "AgentError",
    "CannotHandleTask",
    "ConversationMemory",
    "HFTextGenerationAgent",
    "EchoAgent",
    "EventRecorder",
    "RouterRuntime",
    "RuleAgent",
    "TaskContext",
    "build_agent_from_config",
    "load_config_file",
    "register_agent_builder",
]

