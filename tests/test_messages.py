from mel.mel_bridge import english_to_mel, mel_to_english
from mel.messages import TaskOutput, TaskRequest, TaskResult


def test_task_request_roundtrip():
    req = english_to_mel("Hello world", intent="qa", return_dict=False)
    payload = req.to_dict()
    parsed = TaskRequest.from_dict(payload)
    assert parsed.task.intent == "qa"
    assert parsed.task.inputs[0].value == "Hello world"
    assert parsed.to_dict() == payload


def test_task_result_roundtrip():
    result = TaskResult.ok_result(
        "T-1",
        [TaskOutput(name="answer", kind="text", value="Ready to help!")],
        metrics={"latency_ms": 5},
    )
    payload = result.to_dict()
    parsed = TaskResult.from_dict(payload)
    assert parsed.ok()
    assert parsed.metrics["latency_ms"] == 5
    assert mel_to_english(parsed) == "Ready to help!"
