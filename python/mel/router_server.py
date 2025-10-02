# python/mel/router_server.py
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from mel.mel_validate import validate_obj

REGISTRY = {
    "qa": {"name": "stub.qa.v0", "device": "CPU", "latency_p50_ms": 10},
}

def handle_task(task: dict) -> dict:
    intent = task["intent"]
    if intent == "qa":
        q = task["inputs"][0]["value"].lower()
        ans = "Mount Elbrus, 5,642 m" if "tallest" in q and "europe" in q else "Unknown"
        return {
            "type": "TASK_RESULT",
            "task_id": task["task_id"],
            "status": "ok",
            "outputs": [{"name": "answer", "kind": "text", "value": ans}],
            "metrics": {"latency_ms": 7}
        }
    return {
        "type": "TASK_RESULT",
        "task_id": task["task_id"],
        "status": "error",
        "error": {"code": "UNSUPPORTED_INTENT", "msg": intent}
    }

class Handler(BaseHTTPRequestHandler):
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
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
        except Exception:
            self.send_response(500); self.end_headers()

def main(host="127.0.0.1", port=8089):
    srv = HTTPServer((host, port), Handler)
    print(f"MEL router listening on http://{host}:{port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
