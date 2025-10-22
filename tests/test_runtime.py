from pathlib import Path

from mel.mel_bridge import english_to_mel, mel_to_english
from mel.runtime import RouterRuntime


def test_default_runtime_handles_rule():
    runtime = RouterRuntime.with_defaults()
    req = english_to_mel("What is the tallest mountain in Europe?", intent="qa")
    res = runtime.handle(req)
    assert res.ok()
    assert "Elbrus" in mel_to_english(res)


def test_runtime_from_config(tmp_path):
    config_path = Path(__file__).resolve().parents[1] / "python" / "examples" / "router_config.toml"
    runtime = RouterRuntime.from_config_file(config_path)
    req = english_to_mel("Tell me the weather", intent="chat")
    res = runtime.handle(req)
    assert res.ok()
    answer = mel_to_english(res)
    assert "local demo agent" in answer
