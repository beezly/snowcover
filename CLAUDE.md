# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
# Build Docker image
./build.sh

# Run standalone (no MQTT, console output)
python -m snowcover.main -s -u "rtsp://camera-url/stream"

# Run standalone with different output formats
python -m snowcover.main -s -u "rtsp://..." -f json    # JSON output
python -m snowcover.main -s -u "rtsp://..." -f simple  # One-line output
python -m snowcover.main -s -u "rtsp://..." -f fancy   # ASCII art view
python -m snowcover.main -s -u "rtsp://..." -1         # Single reading, then exit

# Run with config file (service mode with MQTT)
python -m snowcover.main -c config/config.yaml

# Docker compose (uses docker-compose.yaml or docker-compose.local.yaml)
podman compose -f docker-compose.local.yaml up -d
podman logs snowcover-snowcover-1

# Development
pip install -e ".[dev]"
pytest
black src/
mypy src/
```

## Architecture Overview

SnowCover detects snow from RTSP camera feeds and publishes results to Home Assistant via MQTT.

### Core Processing Pipeline

1. **RTSPReader** (`stream/rtsp_reader.py`) - Threaded RTSP capture with reconnection and frame buffering
2. **SnowDetector** (`detection/detector.py`) - Orchestrates detection modules, applies smoothing/debouncing
3. **MQTTPublisher** (`mqtt/publisher.py`) - MQTT client with LWT, subscribes to HA birth messages
4. **HADiscovery** (`mqtt/ha_discovery.py`) - Generates HA MQTT discovery payloads for all entities

### Detection Modules

- **FallingSnowDetector** (`detection/falling_snow.py`) - Frame differencing + particle analysis to detect falling snow
- **IntensityClassifier** (`detection/intensity.py`) - ONNX ML model or particle-density fallback for intensity
- **GroundCoverDetector** (`detection/ground_cover.py`) - HSV analysis (day) or brightness analysis (IR) for ground snow

### Key Design Patterns

- **IR/Night Mode**: Automatically detected via color saturation. All detectors check `is_ir_mode` and adjust thresholds accordingly
- **Smoothing**: Detection results pass through configurable moving average to prevent flicker
- **State Changes**: Only publishes to MQTT on actual state changes or heartbeat interval (60s)
- **HA Birth Messages**: Subscribes to `homeassistant/status` and republishes discovery when HA restarts

### Configuration

Pydantic models in `config.py` with nested environment variable support (`SNOWCOVER_CAMERA__RTSP_URL`, `SNOWCOVER_MQTT__HOST`, etc.)

### Web Server

Optional Flask server (`web/server.py`) provides MJPEG stream at `/stream`, JSON stats at `/stats`, and dashboard at `/`. Enabled via `SNOWCOVER_WEB__ENABLED=true`.

## MQTT Topics

- Discovery: `homeassistant/{sensor,binary_sensor}/snowcover_{camera_id}/{entity}/config`
- State: `snowcover_{camera_id}/{entity}/state`
- Availability: `snowcover_{camera_id}/availability`
