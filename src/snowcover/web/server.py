"""Web server for live video streaming and stats display."""

import threading
import time
from typing import Any, Callable

import cv2
import numpy as np
import structlog
from flask import Flask, Response, jsonify, render_template_string, request

from snowcover.config import WebConfig

logger = structlog.get_logger()

# HTML template for the live view page
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SnowCover - Live View</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
        }
        .container {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            padding: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }
        .video-panel {
            flex: 1;
            min-width: 300px;
        }
        .video-panel img {
            width: 100%;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }
        .stats-panel {
            width: 320px;
            background: #16213e;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }
        h1 {
            text-align: center;
            margin-bottom: 20px;
            color: #4cc9f0;
            font-size: 1.5rem;
        }
        .stat-group {
            margin-bottom: 20px;
        }
        .stat-group h2 {
            font-size: 0.9rem;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
            padding-bottom: 5px;
            border-bottom: 1px solid #333;
        }
        .stat-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
        }
        .stat-label {
            color: #aaa;
        }
        .stat-value {
            font-weight: 600;
            font-family: 'SF Mono', Monaco, monospace;
        }
        .status-snowing {
            color: #4cc9f0;
        }
        .status-ground {
            color: #7209b7;
        }
        .status-clear {
            color: #6c757d;
        }
        .progress-bar {
            width: 100%;
            height: 8px;
            background: #333;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 5px;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4cc9f0, #7209b7);
            transition: width 0.3s ease;
        }
        .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .badge-snowing {
            background: #4cc9f0;
            color: #000;
        }
        .badge-ground {
            background: #7209b7;
            color: #fff;
        }
        .badge-clear {
            background: #333;
            color: #888;
        }
        .timestamp {
            text-align: center;
            color: #666;
            font-size: 0.8rem;
            margin-top: 20px;
        }
        .video-controls {
            margin-top: 10px;
            text-align: center;
        }
        .control-btn {
            background: #16213e;
            color: #4cc9f0;
            border: 1px solid #4cc9f0;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.2s;
        }
        .control-btn:hover {
            background: #4cc9f0;
            color: #16213e;
        }
        .control-btn.active {
            background: #4cc9f0;
            color: #16213e;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="video-panel">
            <img id="video-feed" src="/stream" alt="Live Camera Feed">
            <div class="video-controls">
                <button id="btn-normal" class="control-btn active" onclick="setViewMode('normal')">Normal</button>
                <button id="btn-ground" class="control-btn" onclick="setViewMode('ground')">Ground Region</button>
                <button id="btn-debug" class="control-btn" onclick="setViewMode('debug')">Segmentation</button>
            </div>
        </div>
        <div class="stats-panel">
            <h1>SnowCover</h1>

            <div class="stat-group">
                <h2>Falling Snow</h2>
                <div class="stat-row">
                    <span class="stat-label">Status</span>
                    <span id="snow-status" class="stat-value status-clear">-</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Intensity</span>
                    <span id="intensity-value" class="stat-value">-</span>
                </div>
                <div class="progress-bar">
                    <div id="intensity-bar" class="progress-fill" style="width: 0%"></div>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Particles</span>
                    <span id="particles" class="stat-value">-</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Confidence</span>
                    <span id="snow-confidence" class="stat-value">-</span>
                </div>
            </div>

            <div class="stat-group">
                <h2>Ground Cover</h2>
                <div class="stat-row">
                    <span class="stat-label">Status</span>
                    <span id="ground-status" class="stat-value status-clear">-</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Coverage</span>
                    <span id="coverage-value" class="stat-value">-</span>
                </div>
                <div class="progress-bar">
                    <div id="coverage-bar" class="progress-fill" style="width: 0%"></div>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Brightness</span>
                    <span id="brightness" class="stat-value">-</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Confidence</span>
                    <span id="ground-confidence" class="stat-value">-</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Region</span>
                    <span id="ground-region" class="stat-value">-</span>
                </div>
            </div>

            <div class="stat-group">
                <h2>System</h2>
                <div class="stat-row">
                    <span class="stat-label">Camera Mode</span>
                    <span id="camera-mode" class="stat-value">-</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">CPU Usage</span>
                    <span id="cpu-usage" class="stat-value">-</span>
                </div>
            </div>

            <div class="timestamp">
                Updated: <span id="last-update">-</span>
            </div>
        </div>
    </div>

    <script>
        function updateStats() {
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    // Snow status
                    const snowStatus = document.getElementById('snow-status');
                    if (data.is_snowing) {
                        snowStatus.textContent = data.intensity_category.toUpperCase();
                        snowStatus.className = 'stat-value status-snowing';
                    } else {
                        snowStatus.textContent = 'Clear';
                        snowStatus.className = 'stat-value status-clear';
                    }

                    // Intensity
                    document.getElementById('intensity-value').textContent = data.intensity_percent + '%';
                    document.getElementById('intensity-bar').style.width = data.intensity_percent + '%';
                    document.getElementById('particles').textContent = data.particle_count;
                    document.getElementById('snow-confidence').textContent = Math.round(data.snow_confidence * 100) + '%';

                    // Ground status
                    const groundStatus = document.getElementById('ground-status');
                    if (data.has_ground_snow) {
                        groundStatus.textContent = 'SNOW COVERED';
                        groundStatus.className = 'stat-value status-ground';
                    } else {
                        groundStatus.textContent = 'Clear';
                        groundStatus.className = 'stat-value status-clear';
                    }

                    // Ground coverage
                    document.getElementById('coverage-value').textContent = data.ground_coverage_percent.toFixed(1) + '%';
                    document.getElementById('coverage-bar').style.width = data.ground_coverage_percent + '%';
                    document.getElementById('brightness').textContent = data.ground_brightness.toFixed(1);
                    document.getElementById('ground-confidence').textContent = Math.round(data.ground_confidence * 100) + '%';
                    document.getElementById('ground-region').textContent = data.ground_auto_detected ? 'Auto-detected' : 'Manual';

                    // System
                    document.getElementById('camera-mode').textContent = data.camera_mode === 'ir' ? 'IR / Night' : 'Daylight';
                    document.getElementById('cpu-usage').textContent = data.cpu_usage.toFixed(1) + '%';

                    // Timestamp
                    document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
                })
                .catch(err => console.error('Failed to fetch stats:', err));
        }

        // Update stats every 500ms
        setInterval(updateStats, 500);
        updateStats();

        // View mode control
        let currentMode = 'normal';

        function setViewMode(mode) {
            currentMode = mode;
            const feed = document.getElementById('video-feed');

            // Update feed source
            if (mode === 'normal') {
                feed.src = '/stream';
            } else {
                feed.src = '/stream?overlay=' + mode;
            }

            // Update button states
            document.getElementById('btn-normal').classList.remove('active');
            document.getElementById('btn-ground').classList.remove('active');
            document.getElementById('btn-debug').classList.remove('active');
            document.getElementById('btn-' + mode).classList.add('active');
        }
    </script>
</body>
</html>
"""


class WebServer:
    """Web server for live video streaming and stats display.

    Provides:
    - MJPEG video stream at /stream
    - MJPEG video stream with ground overlay at /stream?overlay=ground
    - JSON stats endpoint at /stats
    - HTML dashboard at /
    """

    def __init__(
        self,
        config: WebConfig,
        get_frame: Callable[[], np.ndarray | None],
        get_stats: Callable[[], dict[str, Any] | None],
        get_overlay_frame: Callable[[], np.ndarray | None] | None = None,
        get_debug_frame: Callable[[], np.ndarray | None] | None = None,
    ):
        """Initialize the web server.

        Args:
            config: Web server configuration
            get_frame: Callback to get current frame
            get_stats: Callback to get current stats
            get_overlay_frame: Optional callback to get frame with ground overlay
            get_debug_frame: Optional callback to get frame with segmentation debug info
        """
        self.config = config
        self._get_frame = get_frame
        self._get_stats = get_stats
        self._get_overlay_frame = get_overlay_frame
        self._get_debug_frame = get_debug_frame
        self._log = logger.bind(component="web_server")

        # Create Flask app
        self._app = Flask(__name__)
        self._app.logger.disabled = True  # Use structlog instead

        # Register routes
        self._app.add_url_rule("/", "index", self._index)
        self._app.add_url_rule("/stream", "stream", self._stream)
        self._app.add_url_rule("/stats", "stats", self._stats)

        self._thread: threading.Thread | None = None
        self._running = False

    def _index(self):
        """Serve the main dashboard page."""
        return render_template_string(HTML_TEMPLATE)

    def _generate_frames(self, mode: str = "normal"):
        """Generate MJPEG frames for streaming.

        Args:
            mode: Frame mode - "normal", "ground", or "debug"
        """
        frame_interval = 1.0 / self.config.frame_rate
        last_frame_time = 0.0

        while self._running:
            current_time = time.time()

            # Rate limiting
            if current_time - last_frame_time < frame_interval:
                time.sleep(0.01)
                continue

            # Get frame based on mode
            if mode == "ground" and self._get_overlay_frame is not None:
                frame = self._get_overlay_frame()
            elif mode == "debug" and self._get_debug_frame is not None:
                frame = self._get_debug_frame()
            else:
                frame = self._get_frame()

            if frame is None:
                time.sleep(0.1)
                continue

            last_frame_time = current_time

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue

            frame_bytes = buffer.tobytes()

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
            )

    def _stream(self):
        """Serve MJPEG video stream."""
        # Check for overlay query parameter
        overlay = request.args.get('overlay', '').lower()
        mode = "normal"
        if overlay == "ground":
            mode = "ground"
        elif overlay == "debug":
            mode = "debug"
        return Response(
            self._generate_frames(mode=mode),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

    def _stats(self):
        """Serve current detection stats as JSON."""
        stats = self._get_stats()
        if stats is None:
            return jsonify({
                "is_snowing": False,
                "intensity_percent": 0,
                "intensity_category": "none",
                "particle_count": 0,
                "snow_confidence": 0,
                "has_ground_snow": False,
                "ground_coverage_percent": 0,
                "ground_confidence": 0,
                "ground_brightness": 0,
                "ground_auto_detected": False,
                "camera_mode": "day",
                "cpu_usage": 0,
            })
        return jsonify(stats)

    def start(self) -> None:
        """Start the web server in a background thread."""
        if self._running:
            return

        self._running = True
        self._log.info(
            "Starting web server",
            host=self.config.host,
            port=self.config.port,
        )

        self._thread = threading.Thread(
            target=self._run_server,
            daemon=True,
        )
        self._thread.start()

    def _run_server(self) -> None:
        """Run the Flask server."""
        # Suppress Flask/Werkzeug logging
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

        self._app.run(
            host=self.config.host,
            port=self.config.port,
            threaded=True,
            use_reloader=False,
        )

    def stop(self) -> None:
        """Stop the web server."""
        self._running = False
        self._log.info("Web server stopped")
