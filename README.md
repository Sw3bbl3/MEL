# MEL - Model Exchange Language

<p align="center">
  <img src="assets/MEL_Small.png" alt="MEL Logo" width="200"/>
</p>

MEL is a compact protocol for on-device multi-model systems. It defines a small, typed message set that lets a router send tasks to models and receive structured results with tight latency and memory budgets.

- Local first
- Deterministic message shapes
- JSON for logs and tests
- Binary planned for runtime
- Tiny adapters so any model can speak MEL

## Quick start

```bash
# Python env
python -m venv .venv
. .venv/Scripts/Activate.ps1  # Windows PowerShell
pip install -U pip
pip install -r requirements.txt
pip install -e ./python

# Validate an example
python -m mel.mel_validate spec/examples/qa_request.json

# Round trip demo
python python/examples/round_trip_demo.py
