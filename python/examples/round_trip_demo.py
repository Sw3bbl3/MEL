import json
from mel.mel_bridge import english_to_mel, mel_to_english
from mel.mel_validate import validate_obj

def fake_oracle(mel_req: dict) -> dict:
    task = mel_req["task"]
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

if __name__ == "__main__":
    req = english_to_mel("What is the tallest mountain in Europe?", intent="qa")
    assert validate_obj(req), "Request failed schema"
    print("MEL request:\n", json.dumps(req, indent=2))
    res = fake_oracle(req)
    print("\nMEL result:\n", json.dumps(res, indent=2))
    text = mel_to_english(res)
    print("\nAnswer:", text)
