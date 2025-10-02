# python/mel/router_client.py
import json, sys, urllib.request
from mel.mel_bridge import english_to_mel

def send(text: str, intent="qa", url="http://127.0.0.1:8089"):
    req = english_to_mel(text, intent=intent)
    body = json.dumps({"type": "TASK_REQUEST", "task": req["task"]}).encode("utf-8")
    r = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    return req, res

if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "What is the tallest mountain in Europe?"
    req, res = send(text)
    print("Request:\n", json.dumps(req, indent=2))
    print("\nResult:\n", json.dumps(res, indent=2))
