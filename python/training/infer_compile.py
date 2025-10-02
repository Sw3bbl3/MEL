# python/training/infer_compile.py
from __future__ import annotations
import json, re
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from mel.mel_bridge import english_to_mel
from mel.mel_validate import validate_obj

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "wave_small_mel_align"
SYS_PREFIX = "<mel_json>"
SYS_SUFFIX = "</mel_json>"

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
    j = s.find("{")
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
    try:
        return json.loads(raw)
    except Exception:
        pass
    s = raw
    s = re.sub(r",\s*([}\]])", r"\1", s)
    s = re.sub(r"}\s*{", "},{", s)
    s = re.sub(r"]\s*{", "],{", s)
    s = re.sub(r'}\s*(")', r'},\1', s)
    s = re.sub(r']\s*(")', r'],\1', s)
    s = re.sub(r'("|\d|true|false|null)\s*(")', r'\1,\2', s)
    try:
        return json.loads(s)
    except Exception:
        s2 = s.strip()
        if not s2.startswith("{"): s2 = "{" + s2
        if not s2.endswith("}"):  s2 = s2 + "}"
        return json.loads(s2)

def _fewshot() -> str:
    # Exemplars help the model, but we never parse them
    ex1_q = "What is the capital of France?"
    ex1 = english_to_mel(ex1_q, intent="qa")
    ex2_q = "Classify: Global markets rallied after the central bank decision."
    ex2 = english_to_mel(ex2_q, intent="classify")
    def fmt(q, mel):
        return f"ENGLISH_TO_MEL:\n{q}\nOUTPUT:\n{SYS_PREFIX}\n{json.dumps(mel, ensure_ascii=False)}\n{SYS_SUFFIX}\n\n"
    return fmt(ex1_q, ex1) + fmt(ex2_q, ex2)

def generate(question: str, intent: str = "qa", max_new_tokens=192):
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Build prompt with few-shot exemplars
    few = _fewshot()
    anchor = f"ENGLISH_TO_MEL:\n{question}\nOUTPUT:\n{SYS_PREFIX}\n"
    prompt = few + anchor

    # Force stop at </mel_json> if that token exists
    suffix_id = tok.convert_tokens_to_ids(SYS_SUFFIX)
    ids = tok(prompt, return_tensors="pt").to(device)

    out = model.generate(
        **ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=(suffix_id if suffix_id is not None else tok.eos_token_id),
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
        no_repeat_ngram_size=3,
    )
    full = tok.decode(out[0], skip_special_tokens=False)

    # Only look after the anchor
    raw = _extract_after_anchor(full, anchor)
    if raw is None:
        # fallback so you always get a usable MEL
        return english_to_mel(question, intent=intent), full, True

    try:
        obj = _json_repair(raw)
    except Exception:
        return english_to_mel(question, intent=intent), full, True

    # Validate shape; if it is not a TASK_REQUEST, fallback
    if not validate_obj(obj):
        return english_to_mel(question, intent=intent), full, True

    return obj, full, False

if __name__ == "__main__":
    q = "What is the tallest mountain in Europe?"
    mel_obj, gen_text, fell_back = generate(q, intent="qa")
    # Print only the query chunk
    print(gen_text[gen_text.rfind(f"ENGLISH_TO_MEL:\n{q}") : ])
    print("Parsed MEL:\n", json.dumps(mel_obj, indent=2))
    print("Valid:", True, "| Fallback:", fell_back)
