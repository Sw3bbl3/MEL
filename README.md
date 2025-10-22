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

### Features

- 💡 **Typed Python models** for building and validating MEL requests/results.
- 🧠 **Configurable multi-agent runtime** with deterministic rule agents, optional Hugging Face generation, and sliding-window conversation memory.
- 🛠️ **Command line tooling** for linting schemas, launching the reference router, and sending ad-hoc tasks.
- 🧾 **Schema-first design** with JSON/TOML configuration support.
- ⚙️ **Extensible agent registry** so integrators can plug in on-device models or custom business logic.

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

### Launch the reference router
```bash
python -m mel.cli serve --host 0.0.0.0 --port 8089
# or provide a custom runtime configuration (JSON/TOML)
python -m mel.cli serve --config python/examples/router_config.toml
```

### Send a request via CLI
```bash
python -m mel.cli send "What is the tallest mountain in Europe?"
```

---

## Repository Structure
```bash
mel/
 ├── python/          # Python package and runtime
 │   ├── mel/         # Library modules, runtime, CLI
 │   └── examples/    # Router configs, integration demos
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
- Richer agent catalog (retrieval-augmented QA, tool calling)
- Streaming transport bindings (gRPC/WebSocket)

---

## Contributing

Contributions are welcome. Please open an issue or pull request with improvements to the specification, reference implementations, or documentation.

---

## License

MEL is released under the [MIT license](LICENSE).

---

<p align="center"> <img src="assets/MEL_Small.png" alt="MEL Logo" width="240"/><br/> <i>Model Exchange Language – a foundation for reliable multi-model systems.</i></p>
<p align="center"> <i>WayV Inc.</i></p>
