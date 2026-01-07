"""Threaded RTSP stream reader with reconnection support."""

import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
import structlog

from snowcover.config import CameraConfig, ProcessingConfig

logger = structlog.get_logger()


@dataclass
class Frame:
    """Container for a captured frame with metadata."""

    data: np.ndarray
    timestamp: float
    frame_number: int


class RTSPReader:
    """Threaded RTSP stream reader that maintains the latest frame.

    Uses a background thread to continuously capture frames, preventing
    buffer backup when processing is slower than the stream frame rate.
    """

    def __init__(
        self,
        camera_config: CameraConfig,
        processing_config: ProcessingConfig,
        on_status_change: Callable[[str], None] | None = None,
    ):
        """Initialize the RTSP reader.

        Args:
            camera_config: Camera/stream configuration
            processing_config: Frame processing configuration
            on_status_change: Optional callback for status changes (online/offline/connecting)
        """
        self.camera_config = camera_config
        self.processing_config = processing_config
        self.on_status_change = on_status_change

        self._running = False
        self._connected = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        self._latest_frame: Frame | None = None
        self._frame_buffer: deque[Frame] = deque(maxlen=processing_config.buffer_size)
        self._frame_count = 0

        self._log = logger.bind(camera_id=camera_config.id)

    @property
    def is_connected(self) -> bool:
        """Check if currently connected to the stream."""
        return self._connected

    @property
    def is_running(self) -> bool:
        """Check if the reader thread is running."""
        return self._running

    def start(self) -> None:
        """Start the background capture thread."""
        if self._running:
            self._log.warning("Reader already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        self._log.info("RTSP reader started")

    def stop(self) -> None:
        """Stop the background capture thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._log.info("RTSP reader stopped")

    def get_frame(self) -> Frame | None:
        """Get the latest captured frame.

        Returns:
            The most recent frame, or None if no frame is available.
        """
        with self._lock:
            return self._latest_frame

    def get_frame_pair(self) -> tuple[Frame, Frame] | None:
        """Get the two most recent frames for temporal analysis.

        Returns:
            Tuple of (previous_frame, current_frame), or None if not enough frames.
        """
        with self._lock:
            if len(self._frame_buffer) < 2:
                return None
            frames = list(self._frame_buffer)
            return frames[-2], frames[-1]

    def get_recent_frames(self, count: int) -> list[Frame]:
        """Get the N most recent frames.

        Args:
            count: Number of frames to retrieve

        Returns:
            List of frames, oldest first. May be shorter than count if
            not enough frames are available.
        """
        with self._lock:
            return list(self._frame_buffer)[-count:]

    def _set_status(self, status: str) -> None:
        """Update status and notify callback."""
        if self.on_status_change:
            self.on_status_change(status)

    def _build_rtsp_url(self) -> str:
        """Build the RTSP URL with transport options."""
        url = self.camera_config.rtsp_url
        transport = self.camera_config.transport.lower()

        # OpenCV uses specific format for transport
        if "?" in url:
            url += f"&rtsp_transport={transport}"
        else:
            url += f"?rtsp_transport={transport}"

        return url

    def _capture_loop(self) -> None:
        """Main capture loop running in background thread."""
        attempts = 0
        frame_interval = 1.0 / self.processing_config.frame_rate
        last_capture_time = 0.0

        while self._running:
            cap = None
            try:
                self._set_status("connecting")
                self._log.info("Connecting to RTSP stream", url=self.camera_config.rtsp_url)

                url = self._build_rtsp_url()
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

                if not cap.isOpened():
                    raise ConnectionError("Failed to open RTSP stream")

                # Configure capture properties
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer to reduce latency

                self._connected = True
                self._set_status("online")
                attempts = 0
                self._log.info("Connected to RTSP stream")

                while self._running and cap.isOpened():
                    # Rate limiting - skip frames to match desired frame rate
                    current_time = time.time()
                    if current_time - last_capture_time < frame_interval:
                        # Grab frame but don't decode (reduces CPU usage)
                        cap.grab()
                        time.sleep(0.01)
                        continue

                    ret, raw_frame = cap.read()
                    if not ret:
                        raise ConnectionError("Failed to read frame from stream")

                    last_capture_time = current_time

                    # Resize frame for processing
                    frame = cv2.resize(
                        raw_frame,
                        (self.processing_config.width, self.processing_config.height),
                    )

                    self._frame_count += 1
                    frame_obj = Frame(
                        data=frame,
                        timestamp=current_time,
                        frame_number=self._frame_count,
                    )

                    with self._lock:
                        self._latest_frame = frame_obj
                        self._frame_buffer.append(frame_obj)

            except Exception as e:
                self._connected = False
                self._set_status("offline")
                self._log.error("Stream error", error=str(e))

                attempts += 1
                max_attempts = self.camera_config.reconnect_max_attempts

                if max_attempts > 0 and attempts >= max_attempts:
                    self._log.critical(
                        "Max reconnection attempts reached",
                        attempts=attempts,
                        max_attempts=max_attempts,
                    )
                    break

                # Exponential backoff with jitter
                base_delay = self.camera_config.reconnect_delay
                delay = min(base_delay * (2 ** min(attempts, 6)), 300)  # Cap at 5 minutes
                jitter = random.uniform(0, delay * 0.1)
                total_delay = delay + jitter

                self._log.info(
                    "Reconnecting",
                    attempt=attempts,
                    delay_seconds=round(total_delay, 1),
                )
                time.sleep(total_delay)

            finally:
                if cap is not None:
                    cap.release()

        self._connected = False
        self._set_status("offline")
        self._log.info("Capture loop ended")

    def __enter__(self) -> "RTSPReader":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()
