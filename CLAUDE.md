# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- Python 3.10, virtual environment at `.venv/`
- Activate: `source .venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`
- Formatter: Black (configured via PyCharm)

## Running

```bash
python main.py
```

## Dependencies & Domain

The `requirements.txt` reveals the project's intended domain — IoT / robotics / embedded systems:

| Package | Purpose |
|---|---|
| `opencv-python` | Computer vision / image processing |
| `numpy` | Numerical computation |
| `pyyaml` | YAML-based configuration files |
| `python-dotenv` | Runtime config via `.env` files |
| `pyserial` | Serial port communication with hardware |
| `transitions` | Finite state machine logic |
| `loguru` | Structured logging |
| `paho-mqtt` | MQTT messaging (IoT broker communication) |

## Hardware

| Device | Interface | Details |
|---|---|---|
| Dobot E6 | TCP/IP | Dashboard-Port `29999`, Motion-Port `30003` |
| Dobot Magician Förderband | Serial | `/dev/ttyUSB0` |
| USB-Kameras (×2) | USB | OpenCV video capture, used for vision tasks |
| SID1 | UDP | Steuerung über UDP-Pakete |
| Microcontroller | WiFi (TCP/UDP) | Wireless-Verbindung |

## Architecture

Planned modular structure:

```
devices/        # One module per hardware device
core/           # BaseDevice, shared abstractions, state machines
orchestration/  # Workflow / sequence coordination across devices
config/         # YAML config files, .env loading
tests/          # Unit and integration tests
```

Every device class lives under `devices/` and inherits from `core.BaseDevice`, which defines the mandatory interface:

```python
class BaseDevice:
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def is_connected(self) -> bool: ...
```

Device-specific logic (command sets, protocol framing, error recovery) stays inside the device module and must not leak into `orchestration/`. The orchestration layer coordinates devices solely through the `BaseDevice` interface and higher-level domain methods.
