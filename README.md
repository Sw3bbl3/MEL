# MEL – Model Exchange Language

<p align="center">
  <img src="assets/MEL_Small.png" alt="MEL Logo" width="480"/>
</p>

<p align="center">
  <b>A compact protocol for orchestrating on-device multi-model systems.</b>
</p>

---

## Overview

MEL defines a minimal, typed message schema that allows a router to dispatch tasks to multiple models and receive structured results with strict latency and memory guarantees.

Key principles:

- Local-first execution
- Deterministic message shapes
- JSON for logging and testing
- Binary planned for production runtime
- Lightweight adapters so any model can integrate with MEL

---

## Quick Start

### Environment Setup
```bash
# Create virtual environment
python -m venv .venv
. .venv/Scripts/Activate.ps1  # on Windows PowerShell

# Install dependencies
pip install -U pip
pip install -r requirements.txt
pip install -e ./python
```

### Validate an Example
```bash
python -m mel.mel_validate spec/examples/qa_request.json
```

### Run the Round Trip Demo
```bash
python python/examples/round_trip_demo.py
```

---

## Repository Structure
```bash
mel/
 ├── python/          # Python package and runtime
 ├── spec/            # MEL JSON schema and protocol examples
 ├── assets/          # Logos, diagrams, visual references
 ├── examples/        # Demo scripts and reference flows
 └── README.md
```

---

## Roadmap

- Binary runtime serialization
- Native adapters in C++ and Rust
- Deeper integration with ML runtimes (PyTorch, TensorFlow, ONNX)
- Extended compliance and benchmark suite

---

## Contributing

Contributions are welcome. Please open an issue or pull request with improvements to the specification, reference implementations, or documentation.

---

## License

MEL is released under the [MIT license](LICENSE).

---

<p align="center"> <img src="assets/MEL_Small.png" alt="MEL Logo" width="240"/><br/> <i>Model Exchange Language – a foundation for reliable multi-model systems.</i></p>
<p align="center"> <i>WayV Inc.</i></p>
