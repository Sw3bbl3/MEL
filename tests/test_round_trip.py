from mel.mel_bridge import english_to_mel, mel_to_english
from mel.mel_validate import validate_obj

def test_round_trip():
    req = english_to_mel("What is the tallest mountain in Europe?", intent="qa")
    assert validate_obj(req)
    fake_result = {
        "type": "TASK_RESULT",
        "task_id": req["task"]["task_id"],
        "status": "ok",
        "outputs": [{"name": "answer", "kind": "text", "value": "Mount Elbrus, 5,642 m"}]
    }
    out = mel_to_english(fake_result)
    assert "Elbrus" in out
