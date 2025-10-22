# tests/test_round_trip.py
import json
import threading
import time
import urllib.request

from mel.router_server import main as start_router  # ← key fix


def _run_router(port: int):
    start_router(host="127.0.0.1", port=port)


def _wait_for_server(port: int, tries: int = 40, delay: float = 0.1):
    # If you added /healthz in the server, ping it; otherwise just wait briefly
    for _ in range(tries):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=0.2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(delay)
    return False


def test_router_qa_roundtrip():
    port = 8099  # avoid collisions with a manually run server
    t = threading.Thread(target=_run_router, args=(port,), daemon=True)
    t.start()
    up = _wait_for_server(port)
    if not up:
        # fallback: give it a short fixed warmup if /healthz is not implemented
        time.sleep(0.5)

    req = {
        "type": "TASK_REQUEST",
        "task": {
            "task_id": "T-test",
            "intent": "qa",
            "inputs": [
                {
                    "name": "text",
                    "kind": "text",
                    "lang": "en",
                    "value": "What is the tallest mountain in Europe?",
                }
            ],
            "expected": [{"name": "answer", "kind": "text"}],
            "constraints": {
                "max_latency_ms": 200,
                "max_tokens": 64,
                "device_pref": ["CPU"],
                "deterministic": True,
            },
            "hints": {},
        },
    }

    request = urllib.request.Request(
        f"http://127.0.0.1:{port}",
        data=json.dumps(req).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=2) as resp:
        assert resp.status == 200
        res = json.loads(resp.read().decode("utf-8"))
    assert res["type"] == "TASK_RESULT"
    assert res["status"] == "ok"
    outs = {o["name"]: o["value"] for o in res["outputs"]}
    assert "Elbrus" in outs["answer"]
