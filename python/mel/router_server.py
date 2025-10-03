from http.server import BaseHTTPRequestHandler, HTTPServer
import json, re, os, threading, time
from collections import deque
from mel.mel_validate import validate_obj
import errno

# ====== Lightweight conversation memory (per-process, single-client) ======
CONTEXT = deque(maxlen=6)  # store last 6 turns (user/assistant lines)

# ====== Tiny rules registry — you can expand this over time ======
REGISTRY = {
    "qa": {"name": "router.gen.v1", "device": "CPU", "latency_p50_ms": 40},
    "chat": {"name": "router.gen.v1", "device": "CPU", "latency_p50_ms": 40},
}

# ====== Lazy HF generator ======
_GEN_LOCK = threading.Lock()
_GEN_PIPE = None

def _get_generator():
    """
    Lazily load a small local model.
    Defaults to TinyLlama 1.1B chat. Runs on CPU; faster with GPU.
    You can override via HF_MODEL env var.
    """
    global _GEN_PIPE
    if _GEN_PIPE is not None:
        return _GEN_PIPE

    with _GEN_LOCK:
        if _GEN_PIPE is not None:
            return _GEN_PIPE

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        model_id = os.environ.get("HF_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        device = 0 if torch.cuda.is_available() else -1
        torch.set_num_threads(max(1, os.cpu_count() // 2))

        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32),
            device_map=("auto" if torch.cuda.is_available() else None)
        )

        _GEN_PIPE = pipeline(
            "text-generation",
            model=model,
            tokenizer=tok,
            device=device
        )
        return _GEN_PIPE

# ====== Helpers ======
def _simple_answer_rules(q_lower: str) -> str | None:
    # demo fact
    if "tallest" in q_lower and "europe" in q_lower:
        return "Mount Elbrus, 5,642 m"

    # capitals
    m = re.search(r"\bcapital\s+of\s+([a-z\s\-]+)\??", q_lower)
    if m:
        country = " ".join(m.group(1).strip().replace("-", " ").split())
        capitals = {
            "france": "Paris",
            "spain": "Madrid",
            "germany": "Berlin",
            "italy": "Rome",
            "united kingdom": "London",
            "uk": "London",
            "england": "London",
            "united states": "Washington, D.C.",
            "usa": "Washington, D.C.",
            "canada": "Ottawa",
            "australia": "Canberra",
            "japan": "Tokyo",
            "china": "Beijing",
            "india": "New Delhi",
            "mexico": "Mexico City",
            "brazil": "Brasília",
            "russia": "Moscow",
            "south africa": "Pretoria (administrative)",
            "egypt": "Cairo",
        }
        return capitals.get(country)

    # small-talk heuristic
    if q_lower in {"hi", "hello", "hey"} or q_lower.startswith(("hi ", "hello ", "hey ")):
        return "Hi! How can I help?"

    return None

def _generate_open(q: str) -> str:
    # Short, helpful, neutral tone. A tiny system prompt helps guide the small model.
    system = (
        "You are an advanced conversational assistant. "
        "Be natural, clear, and engaging. "
        "Give helpful, accurate answers. "
        "Keep your tone friendly but professional. "
        "Support multi-turn conversation smoothly."
    )

    # Pack recent context
    ctx_lines = []
    for role, text in CONTEXT:
        prefix = "User:" if role == "user" else "Assistant:"
        ctx_lines.append(f"{prefix} {text}")

    prompt = system + "\n\n" + "\n".join(ctx_lines + [f"User: {q}", "Assistant:"])

    pipe = _get_generator()
    out = pipe(
        prompt,
        max_new_tokens=192,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=pipe.tokenizer.eos_token_id,
        pad_token_id=pipe.tokenizer.eos_token_id,
    )[0]["generated_text"]

    # Extract the assistant part after the last "Assistant:"
    pos = out.rfind("Assistant:")
    ans = out[pos + len("Assistant:"):].strip() if pos >= 0 else out.strip()
    # keep it compact
    return ans.split("\n\n")[0].strip()

def handle_task(task: dict) -> dict:
    intent = (task.get("intent") or "qa").strip().lower()
    text = (task.get("inputs", [{}])[0].get("value") or "").strip()
    q_lower = text.lower()

    # 1) fast rules
    ruled = _simple_answer_rules(q_lower)
    if ruled:
        answer = ruled
    else:
        # 2) generative fallback (chatty, general knowledge)
        answer = _generate_open(text)

    # update context for multi-turn feel
    CONTEXT.append(("user", text))
    CONTEXT.append(("assistant", answer))

    return {
        "type": "TASK_RESULT",
        "task_id": task.get("task_id", "T-unknown"),
        "status": "ok",
        "outputs": [{"name": "answer", "kind": "text", "value": answer}],
        "metrics": {"latency_ms": 35}  # rough placeholder
    }

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/healthz":
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
        else:
            self.send_response(405); self.end_headers()

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(n)
            req = json.loads(raw.decode("utf-8"))
            if req.get("type") != "TASK_REQUEST":
                self.send_response(400); self.end_headers(); return
            task = req.get("task", {})
            if not validate_obj({"type": "TASK_REQUEST", "task": task}):
                self.send_response(422); self.end_headers(); return

            res = handle_task(task)
            out = json.dumps(res).encode("utf-8")

            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
            except (BrokenPipeError, ConnectionAbortedError, OSError) as e:
                # Client went away; don't try to send another response
                # (silently drop to avoid server traceback)
                return
        except Exception:
            # If we got here, only try to answer if the socket is still alive
            try:
                self.send_response(500); self.end_headers()
            except Exception:
                pass

def main(host="127.0.0.1", port=8089):
    # Warm up model so first request isn't slow
    try:
        pipe = _get_generator()
        _ = pipe("Assistant: Hello", max_new_tokens=4, do_sample=False)  # tiny warmup
        print("Generator warmed up.")
    except Exception as e:
        print(f"Warmup failed (continuing): {e}")

    srv = HTTPServer((host, port), Handler)
    print(f"MEL router listening on http://{host}:{port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    host = os.environ.get("MEL_HOST", "127.0.0.1")
    port = int(os.environ.get("MEL_PORT", "8089"))
    main(host, port)
