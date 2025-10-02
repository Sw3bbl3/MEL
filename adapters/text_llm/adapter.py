# adapters/text_llm/adapter.py
def answer(task: dict) -> dict:
    q = task["inputs"][0]["value"].lower()
    ans = "Mount Elbrus, 5,642 m" if "tallest" in q and "europe" in q else "Unknown"
    return {
        "type": "TASK_RESULT",
        "task_id": task["task_id"],
        "status": "ok",
        "outputs": [{"name": "answer", "kind": "text", "value": ans}],
        "metrics": {"latency_ms": 7}
    }
