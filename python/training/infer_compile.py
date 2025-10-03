# python/training/infer_compile.py
from __future__ import annotations
import json, re
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.logits_process import NoBadWordsLogitsProcessor
from mel.mel_bridge import english_to_mel
from mel.mel_validate import validate_obj
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "wave_small_mel"
SYS_PREFIX = "<mel_json>"
SYS_SUFFIX = "</mel_json>"

ROUTER_URL = "http://127.0.0.1:8089"
ROUTER_TIMEOUT_S = 30  # was 3

def _extract_after_anchor(text: str, anchor: str) -> str | None:
    """Return MEL JSON body that appears after anchor. Do not use exemplars."""
    i = text.rfind(anchor)
    if i < 0:
        return None
    tail = text[i:]  # search space after the anchor only
    m = re.search(r"<mel_json>([\s\S]*?)</mel_json>", tail)
    if m:
        return m.group(1).strip()
    # fallback: first balanced {...} in the tail
    s = tail
    j = s.find("{")  # still OK because we put the '{' immediately after SYS_PREFIX
    if j < 0:
        return None
    depth, in_str, esc = 0, False, False
    for k in range(j, len(s)):
        ch = s[k]
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
            continue
        if ch == '"': in_str = True
        elif ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[j:k+1]
    return None

def _json_repair(raw: str) -> dict:
    s = raw.strip()

    # 1) Keep only up to the last closing brace if there’s trailing garbage
    last_curly = s.rfind("}")
    if last_curly != -1:
        s = s[: last_curly + 1]

    # 2) Replace single quotes with double quotes (common model slip)
    #    This is crude but works well on our constrained output.
    s = s.replace("'", '"')

    # 3) Remove stray commas before closing brackets/braces
    s = re.sub(r",\s*([}\]])", r"\1", s)

    # 4) If JSON objects/arrays were concatenated, insert commas
    s = re.sub(r"}\s*{", "},{", s)
    s = re.sub(r"]\s*{", "],{", s)
    s = re.sub(r'}\s*(")', r'},\1', s)
    s = re.sub(r']\s*(")', r'],\1', s)
    s = re.sub(r'("|\d|true|false|null)\s*(")', r'\1,\2', s)

    # 5) Balance braces/brackets by appending missing closers
    opens_curly  = s.count("{")
    closes_curly = s.count("}")
    if closes_curly < opens_curly:
        s += "}" * (opens_curly - closes_curly)

    opens_sq  = s.count("[")
    closes_sq = s.count("]")
    if closes_sq < opens_sq:
        s += "]" * (opens_sq - closes_sq)

    # 6) Ensure we end up with an object
    s2 = s.strip()
    if not s2.startswith("{"): s2 = "{" + s2
    if not s2.endswith("}"):  s2 = s2 + "}"

    return json.loads(s2)

def _fewshot() -> str:
    # Exemplars help the model, but we never parse them
    # One clean QA exemplar is enough; fewer opportunities to copy junk.
    q = "What is the capital of France?"
    mel = english_to_mel(q, intent="qa")
    return f"ENGLISH_TO_MEL:\n{q}\nOUTPUT:\n{SYS_PREFIX}\n{json.dumps(mel, ensure_ascii=False)}\n{SYS_SUFFIX}\n\n"

# --- replace your _normalize with this ---
def _normalize(obj: dict, *, question: str, intent: str) -> dict | None:
    """
    Make a best-effort TASK_REQUEST that passes MEL schema:
    - Accept minor key drift (task_ID -> task_id; dict intent -> "qa").
    - Strip foreign fields (status, outputs) that belong to TASK_RESULT.
    - Fill defaults for expected/constraints/hints.
    Return None if it's hopeless.
    """
    if not isinstance(obj, dict):
        return None

    # Top-level shape
    t = (obj.get("type") or "").strip().upper()
    if t not in {"TASK_REQUEST", "REQUEST", "TASK"}:
        return None

    task = obj.get("task")
    if not isinstance(task, dict):
        task = {}

    # Map common drift
    if "task_ID" in task and "task_id" not in task:
        task["task_id"] = task.pop("task_ID")
    if "taskId" in task and "task_id" not in task:
        task["task_id"] = task.pop("taskId")

    # Remove fields that belong to TASK_RESULT if the model hallucinated them
    for bad in ("status", "outputs", "metrics", "error"):
        task.pop(bad, None)

    # Intent
    it = task.get("intent")
    if isinstance(it, dict) or not it:
        task["intent"] = intent
    else:
        task["intent"] = str(it)

    # Inputs: if missing or malformed, rebuild a single text input
    inputs = task.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        inputs = [{
            "name": "text", "kind": "text", "lang": "en", "value": question
        }]
    else:
        # ensure there is at least one text value
        fixed = []
        for x in inputs:
            if isinstance(x, dict):
                name = x.get("name") or "text"
                kind = x.get("kind") or "text"
                lang = x.get("lang") or "en"
                val  = x.get("value") if isinstance(x.get("value"), str) else question
                fixed.append({"name": name, "kind": kind, "lang": lang, "value": val})
        inputs = fixed or [{
            "name": "text", "kind": "text", "lang": "en", "value": question
        }]
    task["inputs"] = inputs

    # Expected
    expected = task.get("expected")
    if not isinstance(expected, list) or not expected:
        expected = [{"name": "answer", "kind": "text"}]
    else:
        # prune to minimal valid shape
        cleaned = []
        for e in expected:
            if isinstance(e, dict):
                cleaned.append({"name": e.get("name", "answer"), "kind": e.get("kind", "text")})
        expected = cleaned or [{"name": "answer", "kind": "text"}]
    task["expected"] = expected

    # Constraints
    cons = task.get("constraints")
    if not isinstance(cons, dict):
        cons = {}
    cons.setdefault("max_latency_ms", 200)
    cons.setdefault("max_tokens", 128)
    cons.setdefault("device_pref", ["NPU", "CPU"])
    cons.setdefault("deterministic", True)
    task["constraints"] = cons

    # Hints
    if not isinstance(task.get("hints"), dict):
        task["hints"] = {}

    # Task id
    import uuid, re
    tid = task.get("task_id") or task.get("taskId") or task.get("task_ID")
    if not isinstance(tid, str):
        tid = f"T-{uuid.uuid4().hex[:8]}"
    # sanitize: allow only letters, numbers, dash, underscore
    tid_clean = re.sub(r"[^A-Za-z0-9_-]", "", tid)
    if not tid_clean or not tid_clean.startswith(("T-", "t-")):
        tid_clean = f"T-{uuid.uuid4().hex[:8]}"
    task["task_id"] = tid_clean

    return {"type": "TASK_REQUEST", "task": task}

def generate(question: str, intent: str = "qa", max_new_tokens=120):
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    few = _fewshot()
    forced_head = '{"type":"TASK_REQUEST","task":'
    anchor = f"ENGLISH_TO_MEL:\n{question}\nOUTPUT:\n{SYS_PREFIX}\n"
    prompt = few + anchor + forced_head

    suffix_id = tok.convert_tokens_to_ids(SYS_SUFFIX)
    ids = tok(prompt, return_tensors="pt").to(device)

    out = model.generate(
        **ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # deterministic decoding
        no_repeat_ngram_size=3,
                logits_processor=[
                    NoBadWordsLogitsProcessor(
                        bad_words_ids=[
                            tok.encode("TASK_RESULT", add_special_tokens=False),
                            tok.encode("Task_Result", add_special_tokens=False),
                            tok.encode("TASk_RESULT", add_special_tokens=False),
                            tok.encode("TASK", add_special_tokens=False),
                            tok.encode("_RESULT", add_special_tokens=False),
                            tok.encode("</mel_json><mel_json>", add_special_tokens=False),
                        ],
                        eos_token_id=tok.eos_token_id,
                    )
                ],
        eos_token_id=(suffix_id if suffix_id is not None else tok.eos_token_id),
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    full = tok.decode(out[0], skip_special_tokens=False)

    raw = _extract_after_anchor(full, anchor)
    if raw is None:
        req = english_to_mel(question, intent=intent)
        fell_back = True
    else:
        try:
            obj = _json_repair(raw)
        except Exception:
            obj = None

        obj = _normalize(obj, question=question, intent=intent)
        if not obj or not validate_obj(obj):
            req = english_to_mel(question, intent=intent)
            fell_back = True
        else:
            req = obj
            fell_back = False

    # NEW: call the router
    result = None
    try:
        r = requests.post(ROUTER_URL, json=req, timeout=ROUTER_TIMEOUT_S)
        if r.ok:
            try:
                result = r.json()
            except ValueError:
                result = None
    except requests.RequestException:
        result = None

    return req, result, full, fell_back

if __name__ == "__main__":
    q = "What is the tallest mountain in Europe?"
    req, res, gen_text, fell_back = generate(q, intent="qa")

    print(gen_text[gen_text.rfind(f"ENGLISH_TO_MEL:\n{q}") : ])
    print("Parsed MEL:\n", json.dumps(req, indent=2))
    print("Valid:", validate_obj(req), "| Fallback:", fell_back)

    if res:
        print("\nTASK_RESULT:\n", json.dumps(res, indent=2))
        # convenience print
        outs = {o.get("name"): o.get("value") for o in res.get("outputs", [])}
        if "answer" in outs:
            print("\nAnswer:", outs["answer"])
    else:
        print("\nRouter unavailable or returned no result.")