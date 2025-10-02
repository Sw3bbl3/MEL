import json, uuid
from typing import Dict, Any, Tuple

INTENTS = {"qa", "classify", "summarize", "detect", "generate.image", "generate.audio"}

def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"

def english_to_mel(
    text: str,
    intent: str = "qa",
    max_latency_ms: int = 200,
    max_tokens: int = 128,
    device_pref = ("NPU","CPU"),
    deterministic: bool = True,
) -> Dict[str, Any]:
    if intent not in INTENTS:
        raise ValueError(f"Unsupported intent: {intent}")
    return {
        "type": "TASK_REQUEST",
        "task": {
            "task_id": _new_id("T"),
            "intent": intent,
            "inputs": [
                {"name": "text", "kind": "text", "lang": "en", "value": text.strip()}
            ],
            "expected": [
                {"name": "answer", "kind": "text"}
            ],
            "constraints": {
                "max_latency_ms": max_latency_ms,
                "max_tokens": max_tokens,
                "device_pref": list(device_pref),
                "deterministic": deterministic
            },
            "hints": {}
        }
    }

def mel_to_english(result: Dict[str, Any]) -> str:
    if result.get("type") != "TASK_RESULT":
        return ""
    if result.get("status") != "ok":
        err = result.get("error", {})
        return f"Error: {err.get('code','unknown')} {err.get('msg','')}".strip()
    outs = result.get("outputs", [])
    for o in outs:
        if o.get("name") == "answer" and o.get("kind") == "text":
            return str(o.get("value", "")).strip()
    return ""
