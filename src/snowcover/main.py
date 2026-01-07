"""Main entry point for the snow detection service."""

import argparse
import json
import os
import shutil
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import psutil
import structlog

from snowcover.config import Settings


# Get current process for CPU monitoring
_process = psutil.Process(os.getpid())


def get_cpu_usage() -> float:
    """Get current process CPU usage percentage.

    Returns:
        CPU usage as percentage (0-100+, can exceed 100 on multi-core)
    """
    try:
        return _process.cpu_percent(interval=None)
    except Exception:
        return 0.0


from snowcover.detection import SnowDetector
from snowcover.mqtt import HADiscovery, MQTTPublisher
from snowcover.stream import RTSPReader
from snowcover.web import WebServer


def configure_logging(level: str, log_format: str, quiet: bool = False) -> None:
    """Configure structured logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_format: Log format (json or text)
        quiet: If True, suppress all logging (for standalone mode)
    """
    if quiet:
        # Suppress logging in standalone mode
        structlog.configure(
            processors=[structlog.processors.JSONRenderer()],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(file=open("/dev/null", "w")),
            cache_logger_on_first_use=True,
        )
        return

    processors = [
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


class SnowCoverService:
    """Main service orchestrating all components."""

    def __init__(self, settings: Settings):
        """Initialize the service.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self._running = False
        self._log = structlog.get_logger().bind(component="service")

        # Components (initialized in start())
        self._rtsp_reader: RTSPReader | None = None
        self._mqtt_publisher: MQTTPublisher | None = None
        self._ha_discovery: HADiscovery | None = None
        self._detector: SnowDetector | None = None
        self._web_server: WebServer | None = None

        # State tracking
        self._last_publish_time = 0.0
        self._heartbeat_interval = 60.0  # Publish heartbeat every 60 seconds

        # Current state for web server access
        self._current_frame: np.ndarray | None = None
        self._current_stats: dict | None = None
        self._frame_lock = threading.Lock()

    def _on_stream_status_change(self, status: str) -> None:
        """Handle stream status changes.

        Args:
            status: New status (online/offline/connecting)
        """
        self._log.info("Stream status changed", status=status)
        if self._ha_discovery:
            self._ha_discovery.publish_state("status", status)

    def _on_ha_birth(self) -> None:
        """Handle Home Assistant birth message (HA restarted)."""
        self._log.info("Republishing discovery after HA restart")
        if self._ha_discovery:
            self._ha_discovery.publish_discovery()
            self._mqtt_publisher.publish_availability("online")

    def _get_current_frame(self) -> np.ndarray | None:
        """Get current frame for web server."""
        # Get live frame directly from RTSP reader for smooth video
        if self._rtsp_reader:
            frame = self._rtsp_reader.get_frame()
            if frame is not None:
                return frame.data
        return None

    def _get_overlay_frame(self) -> np.ndarray | None:
        """Get current frame with ground region overlay for web server."""
        frame = self._get_current_frame()
        if frame is None:
            return None

        # Apply ground region overlay if detector is available
        if self._detector is not None:
            ground_detector = self._detector.ground_cover_detector
            if ground_detector is not None:
                return ground_detector.get_overlay_image(frame)

        return frame

    def _get_debug_frame(self) -> np.ndarray | None:
        """Get current frame with segmentation debug overlay for web server."""
        frame = self._get_current_frame()
        if frame is None:
            return None

        # Apply debug overlay if segmenter is available
        if self._detector is not None:
            ground_detector = self._detector.ground_cover_detector
            if ground_detector is not None and ground_detector._segmenter is not None:
                return ground_detector._segmenter.get_debug_overlay(frame)

        return frame

    def _get_current_stats(self) -> dict | None:
        """Get current stats for web server."""
        with self._frame_lock:
            return self._current_stats

    def start(self) -> None:
        """Start all service components."""
        self._log.info("Starting SnowCover service")

        # Initialize CPU monitoring (first call establishes baseline)
        get_cpu_usage()

        # Initialize MQTT
        self._mqtt_publisher = MQTTPublisher(
            config=self.settings.mqtt,
            camera_id=self.settings.camera.id,
        )
        self._mqtt_publisher.connect()

        # Initialize HA discovery
        self._ha_discovery = HADiscovery(
            publisher=self._mqtt_publisher,
            camera_id=self.settings.camera.id,
        )
        self._ha_discovery.publish_discovery()

        # Set up callback to republish discovery when HA restarts
        self._mqtt_publisher.set_on_ha_birth(self._on_ha_birth)

        # Initialize detector
        self._detector = SnowDetector(
            detection_config=self.settings.detection,
            smoothing_config=self.settings.smoothing,
        )

        # Initialize RTSP reader
        # Use higher capture rate when web is enabled for smooth video
        processing_config = self.settings.processing
        if self.settings.web.enabled:
            from snowcover.config import ProcessingConfig
            processing_config = ProcessingConfig(
                frame_rate=max(self.settings.web.frame_rate, self.settings.processing.frame_rate),
                width=self.settings.processing.width,
                height=self.settings.processing.height,
                buffer_size=self.settings.processing.buffer_size,
            )
            self._log.info("Web enabled, increased capture rate", capture_fps=processing_config.frame_rate)

        self._rtsp_reader = RTSPReader(
            camera_config=self.settings.camera,
            processing_config=processing_config,
            on_status_change=self._on_stream_status_change,
        )
        self._rtsp_reader.start()

        # Initialize web server if enabled
        if self.settings.web.enabled:
            self._web_server = WebServer(
                config=self.settings.web,
                get_frame=self._get_current_frame,
                get_stats=self._get_current_stats,
                get_overlay_frame=self._get_overlay_frame,
                get_debug_frame=self._get_debug_frame,
            )
            self._web_server.start()
            self._log.info(
                "Web server started",
                url=f"http://{self.settings.web.host}:{self.settings.web.port}",
            )

        self._running = True
        self._log.info("SnowCover service started")

    def stop(self) -> None:
        """Stop all service components."""
        self._log.info("Stopping SnowCover service")
        self._running = False

        if self._web_server:
            self._web_server.stop()

        if self._rtsp_reader:
            self._rtsp_reader.stop()

        if self._ha_discovery:
            self._ha_discovery.publish_state("status", "offline")

        if self._mqtt_publisher:
            self._mqtt_publisher.disconnect()

        self._log.info("SnowCover service stopped")

    def run(self) -> None:
        """Main processing loop."""
        if not self._running:
            raise RuntimeError("Service not started")

        frame_interval = 1.0 / self.settings.processing.frame_rate
        last_process_time = 0.0

        self._log.info(
            "Starting processing loop",
            frame_rate=self.settings.processing.frame_rate,
        )

        while self._running:
            try:
                current_time = time.time()

                # Rate limiting
                if current_time - last_process_time < frame_interval:
                    time.sleep(0.05)
                    continue

                # Get frames for processing
                frame_pair = self._rtsp_reader.get_frame_pair()
                if frame_pair is None:
                    # Not enough frames yet
                    time.sleep(0.1)
                    continue

                previous_frame, current_frame = frame_pair
                last_process_time = current_time

                # Run detection
                result = self._detector.detect(
                    current_frame=current_frame.data,
                    previous_frame=previous_frame.data,
                )

                # Get smoothed result
                smoothed = self._detector.get_smoothed_result()
                if smoothed is None:
                    self._log.debug("No smoothed result yet, waiting for more samples")
                    continue

                # Determine IR mode
                is_ir_mode = (
                    result.raw_falling_snow is not None
                    and result.raw_falling_snow.is_ir_mode
                )

                # Update state for web server
                if self._web_server:
                    # Check if ground was auto-detected
                    ground_auto_detected = False
                    ground_detector = self._detector.ground_cover_detector
                    if ground_detector is not None:
                        ground_auto_detected = ground_detector.is_auto_detected

                    with self._frame_lock:
                        self._current_frame = current_frame.data.copy()
                        self._current_stats = {
                            "is_snowing": smoothed.is_snowing,
                            "intensity_percent": smoothed.intensity_percent,
                            "intensity_category": smoothed.intensity_category,
                            "particle_count": smoothed.particle_count,
                            "snow_confidence": smoothed.snow_confidence,
                            "has_ground_snow": smoothed.has_ground_snow,
                            "ground_coverage_percent": smoothed.ground_coverage_percent,
                            "ground_confidence": smoothed.ground_confidence,
                            "ground_brightness": smoothed.ground_brightness,
                            "ground_auto_detected": ground_auto_detected,
                            "camera_mode": "ir" if is_ir_mode else "day",
                            "cpu_usage": get_cpu_usage(),
                        }

                # Check for state changes
                changes = self._detector.get_state_changes(smoothed)

                # Publish changes
                if changes or (current_time - self._last_publish_time >= self._heartbeat_interval):
                    self._log.debug(
                        "Publishing state",
                        is_snowing=smoothed.is_snowing,
                        intensity=smoothed.intensity_percent,
                        has_changes=bool(changes),
                    )
                    self._publish_state(smoothed, is_ir_mode)
                    self._last_publish_time = current_time

            except Exception as e:
                self._log.error("Processing error", error=str(e))
                time.sleep(1.0)

    def _publish_state(self, result, is_ir_mode: bool) -> None:
        """Publish detection state to Home Assistant.

        Args:
            result: Detection result
            is_ir_mode: Whether camera is in IR mode
        """
        if not self._ha_discovery:
            return

        now = datetime.now(timezone.utc).isoformat()

        # Publish binary sensors
        self._ha_discovery.publish_state("snowing", result.is_snowing)
        self._ha_discovery.publish_state("ground_snow", result.has_ground_snow)

        # Publish sensors
        self._ha_discovery.publish_state("particle_count", result.particle_count)
        self._ha_discovery.publish_state("snow_confidence", round(result.snow_confidence * 100, 1))
        self._ha_discovery.publish_state("intensity", result.intensity_percent)
        self._ha_discovery.publish_state("intensity_text", result.intensity_category)
        self._ha_discovery.publish_state("ground_cover", result.ground_coverage_percent)
        self._ha_discovery.publish_state("ground_confidence", round(result.ground_confidence * 100, 1))
        self._ha_discovery.publish_state("ground_brightness", round(result.ground_brightness, 1))
        self._ha_discovery.publish_state("camera_mode", "IR/Night" if is_ir_mode else "Day")
        self._ha_discovery.publish_state("cpu_usage", round(get_cpu_usage(), 1))

        # Publish attributes for main sensors
        self._ha_discovery.publish_attributes("snowing", {
            "last_update": now,
            "confidence": round(result.confidence, 2),
            "particle_count": result.particle_count,
        })

        self._ha_discovery.publish_attributes("intensity", {
            "last_update": now,
            "category": result.intensity_category,
            "model_available": result.model_available,
        })

        self._ha_discovery.publish_attributes("ground_cover", {
            "last_update": now,
            "threshold": self.settings.detection.ground_cover.coverage_threshold,
        })


class StandaloneRunner:
    """Standalone runner that outputs detection results to console."""

    def __init__(self, settings: Settings, output_format: str = "table", continuous: bool = True):
        """Initialize standalone runner.

        Args:
            settings: Application settings
            output_format: Output format (table, json, simple)
            continuous: If True, run continuously; if False, output once and exit
        """
        self.settings = settings
        self.output_format = output_format
        self.continuous = continuous
        self._running = False

        self._rtsp_reader: RTSPReader | None = None
        self._detector: SnowDetector | None = None

    def start(self) -> None:
        """Start the standalone runner."""
        # Initialize CPU monitoring (first call establishes baseline)
        get_cpu_usage()

        # Initialize detector
        self._detector = SnowDetector(
            detection_config=self.settings.detection,
            smoothing_config=self.settings.smoothing,
        )

        # For fancy mode, use higher frame rate for smooth video
        processing_config = self.settings.processing
        if self.output_format == "fancy":
            from snowcover.config import ProcessingConfig
            processing_config = ProcessingConfig(
                frame_rate=30.0,  # High capture rate for smooth video
                width=self.settings.processing.width,
                height=self.settings.processing.height,
                buffer_size=self.settings.processing.buffer_size,
            )

        # Initialize RTSP reader (no status callback needed)
        self._rtsp_reader = RTSPReader(
            camera_config=self.settings.camera,
            processing_config=processing_config,
        )
        self._rtsp_reader.start()
        self._running = True

        if self.output_format == "table" and self.continuous:
            # Print header for continuous table output
            self._print_header()

    def stop(self) -> None:
        """Stop the standalone runner."""
        self._running = False
        if self._rtsp_reader:
            self._rtsp_reader.stop()

    def _print_header(self) -> None:
        """Print table header."""
        print("\n" + "=" * 110)
        print(f"{'Time':<12} {'Snowing':<10} {'Intensity':<12} {'Ground':<10} {'Coverage':<10} {'Conf':<8} {'Particles':<10} {'Mode':<6} {'CPU':<8}")
        print("=" * 110)

    def _format_table(self, result, is_ir_mode: bool, cpu_usage: float) -> str:
        """Format result as table row."""
        now = datetime.now().strftime("%H:%M:%S")
        snowing = "YES" if result.is_snowing else "no"
        intensity = f"{result.intensity_percent}% ({result.intensity_category})"
        ground = "YES" if result.has_ground_snow else "no"
        coverage = f"{result.ground_coverage_percent:.1f}%"
        conf = f"{int(result.confidence * 100)}%"
        particles = str(result.particle_count)
        mode = "IR" if is_ir_mode else "Day"
        cpu = f"{cpu_usage:.1f}%"

        # Color coding for terminal (ANSI)
        if result.is_snowing:
            snowing = f"\033[96m{snowing}\033[0m"  # Cyan
        if result.has_ground_snow:
            ground = f"\033[94m{ground}\033[0m"  # Blue

        return f"{now:<12} {snowing:<10} {intensity:<12} {ground:<10} {coverage:<10} {conf:<8} {particles:<10} {mode:<6} {cpu:<8}"

    def _format_simple(self, result, is_ir_mode: bool, cpu_usage: float) -> str:
        """Format result as simple one-line output."""
        now = datetime.now().strftime("%H:%M:%S")
        parts = [f"[{now}]"]

        snow_conf = int(result.snow_confidence * 100)
        ground_conf = int(result.ground_confidence * 100)

        if result.is_snowing:
            parts.append(f"SNOWING ({result.intensity_category}, {result.intensity_percent}%) [conf:{snow_conf}%]")
        else:
            parts.append(f"Not snowing [conf:{snow_conf}%]")

        if result.has_ground_snow:
            parts.append(f"Ground: {result.ground_coverage_percent:.1f}% covered [conf:{ground_conf}%]")
        else:
            parts.append(f"Ground: clear [conf:{ground_conf}%]")

        parts.append(f"[{'IR' if is_ir_mode else 'Day'}]")
        parts.append(f"CPU: {cpu_usage:.1f}%")

        return " | ".join(parts)

    def _format_json(self, result, is_ir_mode: bool, cpu_usage: float) -> str:
        """Format result as JSON."""
        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "is_snowing": result.is_snowing,
            "particle_count": result.particle_count,
            "snow_confidence": round(result.snow_confidence, 3),
            "intensity_percent": result.intensity_percent,
            "intensity_category": result.intensity_category,
            "has_ground_snow": result.has_ground_snow,
            "ground_coverage_percent": result.ground_coverage_percent,
            "ground_confidence": round(result.ground_confidence, 3),
            "ground_brightness": round(result.ground_brightness, 1),
            "confidence": round(result.confidence, 3),
            "camera_mode": "ir" if is_ir_mode else "day",
            "cpu_usage": round(cpu_usage, 1),
        }
        return json.dumps(data)

    def _enhance_frame(self, gray: np.ndarray) -> np.ndarray:
        """Enhance frame for better ASCII art visibility.

        Applies:
        1. CLAHE for local contrast enhancement
        2. Unsharp masking for edge definition
        3. Gamma correction to spread midtones

        Args:
            gray: Grayscale image

        Returns:
            Enhanced grayscale image
        """
        # 1. CLAHE - Contrast Limited Adaptive Histogram Equalization
        # Enhances local contrast without over-amplifying noise
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 2. Unsharp masking - enhances edges
        # Create blurred version and subtract to get edges
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        unsharp = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

        # 3. Gamma correction - spread out midtones (gamma < 1 brightens midtones)
        gamma = 0.85
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        corrected = cv2.LUT(unsharp, table)

        return corrected

    def _frame_to_ascii(self, frame: np.ndarray, width: int = 80, height: int = 24) -> str:
        """Convert a frame to ASCII art.

        Args:
            frame: BGR frame
            width: Output width in characters
            height: Output height in characters

        Returns:
            ASCII art string
        """
        # Extended ASCII characters from dark to light for better gradients
        ascii_chars = " `.-':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@"

        # Terminal characters are ~2x taller than wide, so we sample at 2x width
        # and average pairs of pixels for better aspect ratio
        sample_width = width * 2
        resized = cv2.resize(frame, (sample_width, height), interpolation=cv2.INTER_AREA)

        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Apply enhancement for better contrast and edge visibility
        enhanced = self._enhance_frame(gray)

        # Map pixel values to ASCII characters
        lines = []
        for row in enhanced:
            line = ""
            # Average pairs of horizontal pixels for correct aspect ratio
            for i in range(0, len(row) - 1, 2):
                pixel = (int(row[i]) + int(row[i + 1])) // 2
                # Map 0-255 to index in ascii_chars
                idx = int(pixel / 256 * len(ascii_chars))
                idx = min(idx, len(ascii_chars) - 1)
                line += ascii_chars[idx]
            lines.append(line)

        return "\n".join(lines)

    def _format_fancy(
        self, result, frame: np.ndarray, is_ir_mode: bool, cpu_usage: float
    ) -> str:
        """Format result with ASCII art view and stats panel."""
        now = datetime.now().strftime("%H:%M:%S")

        # Get terminal size
        try:
            term_size = shutil.get_terminal_size((100, 30))
            term_width = term_size.columns
            term_height = term_size.lines
        except Exception:
            term_width, term_height = 100, 30

        # Stats panel is 30 chars wide, leave space for borders and gap
        stats_width = 30
        ascii_width = max(40, term_width - stats_width - 5)  # 5 for borders and space
        ascii_height = max(10, term_height - 4)  # 4 for top/bottom borders and margin

        # Generate ASCII art scaled to terminal
        ascii_art = self._frame_to_ascii(frame, width=ascii_width, height=ascii_height)
        ascii_lines = ascii_art.split("\n")

        # Build stats panel (fixed width of 28 inner + 2 border = 30)
        # Helper to create a padded line with optional color
        def stat_line(label: str, value: str, color: str = "") -> str:
            reset = "\033[0m" if color else ""
            # Calculate visible width (26 chars inner content)
            visible = f" {label}{value} "
            padding = 26 - len(f" {label}{value} ")
            if padding < 0:
                padding = 0
            return f"│ {label}{color}{value}{reset}{' ' * padding} │"

        CYAN = "\033[96m"
        BLUE = "\033[94m"
        RESET = "\033[0m"

        # Snow status line
        if result.is_snowing:
            snow_line = f"│ Status: {CYAN}SNOWING{RESET}              │"
        else:
            snow_line = f"│ Status: Not snowing        │"

        # Ground status line
        if result.has_ground_snow:
            ground_line = f"│ Status: {BLUE}SNOW COVERED{RESET}         │"
        else:
            ground_line = f"│ Status: Clear              │"

        mode_str = "IR/Night" if is_ir_mode else "Daylight"
        intensity_str = f"{result.intensity_percent:>3}% ({result.intensity_category:<8})"

        stats = [
            f"┌{'─' * 28}┐",
            f"│ {'SnowCover':^26} │",
            f"│ {now:^26} │",
            f"├{'─' * 28}┤",
            f"│ Mode: {mode_str:<20} │",
            f"├{'─' * 28}┤",
            f"│ {'FALLING SNOW':^26} │",
            snow_line,
            f"│ Intensity: {intensity_str:<15} │",
            f"│ Particles: {result.particle_count:>14} │",
            f"│ Confidence: {int(result.snow_confidence * 100):>13}% │",
            f"├{'─' * 28}┤",
            f"│ {'GROUND COVER':^26} │",
            ground_line,
            f"│ Coverage: {result.ground_coverage_percent:>15.1f}% │",
            f"│ Brightness: {result.ground_brightness:>13.1f} │",
            f"│ Confidence: {int(result.ground_confidence * 100):>13}% │",
            f"├{'─' * 28}┤",
            f"│ CPU: {cpu_usage:>21.1f}% │",
            f"└{'─' * 28}┘",
        ]

        # Combine ASCII art with stats panel side by side
        output_lines = []

        # Clear screen and move cursor to top
        output_lines.append("\033[2J\033[H")

        # Top border for ASCII art
        output_lines.append(f"┌{'─' * ascii_width}┐")

        # Combine each line
        for i in range(max(len(ascii_lines), len(stats))):
            ascii_line = ascii_lines[i] if i < len(ascii_lines) else " " * ascii_width
            # Pad ASCII line to correct width
            ascii_line = ascii_line.ljust(ascii_width)[:ascii_width]
            stats_line = stats[i] if i < len(stats) else ""
            output_lines.append(f"│{ascii_line}│ {stats_line}")

        # Bottom border for ASCII art
        output_lines.append(f"└{'─' * ascii_width}┘")

        return "\n".join(output_lines)

    def run(self) -> None:
        """Main processing loop."""
        if not self._running:
            raise RuntimeError("Runner not started")

        # Frame rate settings
        if self.output_format == "fancy":
            display_interval = 1.0 / 24.0  # 24 FPS display
            detection_interval = 0.5  # Run detection every 0.5s
        else:
            display_interval = 1.0 / self.settings.processing.frame_rate
            detection_interval = display_interval  # Same as display

        last_display_time = 0.0
        last_detection_time = 0.0
        cached_result = None
        cached_is_ir_mode = False
        cached_cpu_usage = 0.0

        # Wait for stream to connect
        print("Connecting to camera...", end="", flush=True)
        wait_start = time.time()
        while self._running and not self._rtsp_reader.is_connected:
            if time.time() - wait_start > 30:
                print(" TIMEOUT")
                return
            time.sleep(0.5)
            print(".", end="", flush=True)
        print(" connected!")

        # Wait for enough frames
        print("Buffering frames...", end="", flush=True)
        while self._running:
            frame_pair = self._rtsp_reader.get_frame_pair()
            if frame_pair is not None:
                print(" ready!")
                break
            time.sleep(0.2)
            print(".", end="", flush=True)

        while self._running:
            try:
                current_time = time.time()

                # Rate limiting for display
                if current_time - last_display_time < display_interval:
                    time.sleep(0.01)  # Short sleep for responsive display
                    continue

                # Get frames
                frame_pair = self._rtsp_reader.get_frame_pair()
                if frame_pair is None:
                    time.sleep(0.01)
                    continue

                previous_frame, current_frame = frame_pair
                last_display_time = current_time

                # Run detection less frequently
                if current_time - last_detection_time >= detection_interval:
                    result = self._detector.detect(
                        current_frame=current_frame.data,
                        previous_frame=previous_frame.data,
                    )

                    smoothed = self._detector.get_smoothed_result()
                    if smoothed is not None:
                        cached_result = smoothed
                        cached_is_ir_mode = (
                            result.raw_falling_snow is not None
                            and result.raw_falling_snow.is_ir_mode
                        )
                        cached_cpu_usage = get_cpu_usage()

                    last_detection_time = current_time

                # Skip output if no detection result yet
                if cached_result is None:
                    continue

                # Output result
                if self.output_format == "json":
                    print(self._format_json(cached_result, cached_is_ir_mode, cached_cpu_usage))
                elif self.output_format == "simple":
                    print(self._format_simple(cached_result, cached_is_ir_mode, cached_cpu_usage))
                elif self.output_format == "fancy":
                    # Fancy mode uses live frame with cached detection result
                    print(self._format_fancy(cached_result, current_frame.data, cached_is_ir_mode, cached_cpu_usage))
                else:  # table
                    print(self._format_table(cached_result, cached_is_ir_mode, cached_cpu_usage))

                # Exit after one output if not continuous
                if not self.continuous:
                    break

            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                time.sleep(1.0)

    def __enter__(self) -> "StandaloneRunner":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


def main() -> int:
    """Main entry point.

    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="SnowCover - ML-based snow detection from RTSP camera feeds"
    )
    parser.add_argument(
        "-c", "--config",
        type=Path,
        help="Path to configuration file (YAML)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Override log level from config",
    )

    # Standalone mode arguments
    parser.add_argument(
        "-s", "--standalone",
        action="store_true",
        help="Run in standalone mode (no MQTT, output to console)",
    )
    parser.add_argument(
        "-u", "--url",
        type=str,
        help="RTSP URL (standalone mode, overrides config)",
    )
    parser.add_argument(
        "-f", "--format",
        choices=["table", "json", "simple", "fancy"],
        default="table",
        help="Output format for standalone mode (default: table)",
    )
    parser.add_argument(
        "-1", "--once",
        action="store_true",
        help="Output once and exit (standalone mode only)",
    )

    args = parser.parse_args()

    # Load configuration
    try:
        # Allow standalone mode without config if URL is provided
        if args.standalone and args.url and not args.config:
            from snowcover.config import CameraConfig
            settings = Settings(camera=CameraConfig(rtsp_url=args.url))
        else:
            settings = Settings.load(config_path=args.config)

        # Override RTSP URL if provided
        if args.url:
            settings.camera.rtsp_url = args.url

    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        return 1

    # Standalone mode
    if args.standalone:
        configure_logging("WARNING", "text", quiet=True)

        runner = StandaloneRunner(
            settings=settings,
            output_format=args.format,
            continuous=not args.once,
        )

        def signal_handler(signum, frame):
            print("\nShutting down...")
            runner.stop()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            runner.start()
            runner.run()
        except KeyboardInterrupt:
            pass
        finally:
            runner.stop()

        return 0

    # Normal service mode with MQTT
    log_level = args.log_level or settings.logging.level
    configure_logging(log_level, settings.logging.format)

    logger = structlog.get_logger()
    logger.info(
        "Configuration loaded",
        camera_id=settings.camera.id,
        rtsp_url=settings.camera.rtsp_url[:50] + "..." if len(settings.camera.rtsp_url) > 50 else settings.camera.rtsp_url,
    )

    service = SnowCoverService(settings)

    def signal_handler(signum, frame):
        logger.info("Received shutdown signal", signal=signum)
        service.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        service.start()
        service.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error("Fatal error", error=str(e))
        return 1
    finally:
        service.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
