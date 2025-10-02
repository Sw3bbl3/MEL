# tests/test_router_echo.py
import threading
import time
import json
import urllib.request

from mel.mel_bridge import english_to_mel
from mel.mel_validate import validate_obj
from mel.router_server import main as router_main  # <- fixed import

def _start_router():
    t = threading.Thread(
        target=router_main,
        kwargs={"host": "127.0.0.1", "port": 8091},
        daemon=True,
    )
    t.start()
    time.sleep(0.3)  # give the server a moment to bind
    return t

def test_router_round_trip():
    _start_router()
    req = english_to_mel("What is the tallest mountain in Europe?", intent="qa")
    assert validate_obj({"type": "TASK_REQUEST", "task": req["task"]})

    body = json.dumps({"type": "TASK_REQUEST", "task": req["task"]}).encode("utf-8")
    r = urllib.request.Request(
        "http://127.0.0.1:8091",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(r, timeout=5) as resp:
        assert resp.status == 200
        res = json.loads(resp.read().decode("utf-8"))

    assert res["status"] == "ok"
    assert "Elbrus" in res["outputs"][0]["value"]
