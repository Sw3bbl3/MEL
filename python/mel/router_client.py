# python/mel/router_client.py
import json
import sys
import urllib.request
from typing import Optional, Tuple

from .mel_bridge import english_to_mel, mel_to_english
from .messages import TaskRequest, TaskResult


def send(
    text: str,
    *,
    intent: str = "qa",
    url: str = "http://127.0.0.1:8089",
    session_id: Optional[str] = None,
) -> Tuple[TaskRequest, TaskResult]:
    request = english_to_mel(text, intent=intent, return_dict=False)
    if session_id:
        request.task.hints["session_id"] = session_id
    body = json.dumps(request.to_dict()).encode("utf-8")
    r = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r) as resp:
        result_payload = json.loads(resp.read().decode("utf-8"))
    result = TaskResult.from_dict(result_payload)
    return request, result


def _main() -> int:
    text = " ".join(sys.argv[1:]) or "What is the tallest mountain in Europe?"
    req, res = send(text)
    print("Request:\n", json.dumps(req.to_dict(), indent=2))
    print("\nResult:\n", json.dumps(res.to_dict(), indent=2))
    if res.ok():
        print("\nAnswer:", mel_to_english(res))
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover - manual usage entrypoint
    sys.exit(_main())
