"""Configuration management with Pydantic validation."""

import os
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class CameraConfig(BaseModel):
    """Camera/RTSP stream configuration."""

    id: str = Field(default="camera", description="Unique identifier for this camera")
    rtsp_url: str = Field(..., description="RTSP stream URL")
    transport: str = Field(default="tcp", description="Transport protocol (tcp/udp)")
    reconnect_delay: float = Field(default=5.0, description="Seconds between reconnection attempts")
    reconnect_max_attempts: int = Field(
        default=0, description="Max reconnection attempts (0 = infinite)"
    )


class ProcessingConfig(BaseModel):
    """Frame processing configuration."""

    frame_rate: float = Field(default=1.0, description="Frames per second to process")
    width: int = Field(default=640, description="Frame width for processing")
    height: int = Field(default=480, description="Frame height for processing")
    buffer_size: int = Field(default=10, description="Number of frames to keep in buffer")


class FallingSnowConfig(BaseModel):
    """Falling snow detection configuration."""

    enabled: bool = Field(default=True)
    min_particle_size: int = Field(default=2, description="Minimum pixel size for snow particle")
    max_particle_size: int = Field(default=20, description="Maximum pixel size for snow particle")
    min_particle_count: int = Field(
        default=50, description="Minimum particles to trigger detection"
    )
    brightness_threshold: int = Field(
        default=200, description="Minimum brightness for snow (0-255)"
    )
    motion_threshold: int = Field(default=25, description="Pixel difference threshold")

    # IR/night mode adjustments
    ir_brightness_threshold: int = Field(
        default=180, description="Brightness threshold for IR mode (typically lower)"
    )
    ir_motion_threshold: int = Field(
        default=20, description="Motion threshold for IR mode (more sensitive)"
    )


class IntensityConfig(BaseModel):
    """Snow intensity classification configuration."""

    enabled: bool = Field(default=True)
    model_path: str = Field(
        default="models/weather_classifier.onnx", description="Path to ONNX model"
    )
    light_threshold: int = Field(default=10, description="Threshold for light snow (%)")
    moderate_threshold: int = Field(default=40, description="Threshold for moderate snow (%)")
    heavy_threshold: int = Field(default=70, description="Threshold for heavy snow (%)")


class GroundCoverConfig(BaseModel):
    """Ground snow cover detection configuration."""

    enabled: bool = Field(default=True)
    auto_detect: bool = Field(
        default=True,
        description="Automatically detect ground region using ML segmentation",
    )
    segmentation_model: str = Field(
        default="models/ground_segmentation.onnx",
        description="Path to ground segmentation ONNX model",
    )
    ground_region: list[list[float]] = Field(
        default=[[0.0, 0.6], [1.0, 0.6], [1.0, 1.0], [0.0, 1.0]],
        description="Ground region polygon as percentage of frame (fallback if auto_detect fails)",
    )
    snow_hsv_lower: list[int] = Field(
        default=[0, 0, 200], description="Lower HSV bounds for snow detection"
    )
    snow_hsv_upper: list[int] = Field(
        default=[180, 50, 255], description="Upper HSV bounds for snow detection"
    )
    coverage_threshold: int = Field(
        default=30, description="% coverage to trigger binary sensor"
    )

    # IR/night mode settings
    ir_brightness_threshold: int = Field(
        default=180, description="Brightness threshold for snow in IR mode"
    )
    ir_saturation_threshold: float = Field(
        default=15.0, description="Max saturation to detect IR mode"
    )


class DetectionConfig(BaseModel):
    """Detection module configuration."""

    falling_snow: FallingSnowConfig = Field(default_factory=FallingSnowConfig)
    intensity: IntensityConfig = Field(default_factory=IntensityConfig)
    ground_cover: GroundCoverConfig = Field(default_factory=GroundCoverConfig)


class SmoothingConfig(BaseModel):
    """Result smoothing and debouncing configuration."""

    window_size: int = Field(default=5, description="Number of readings to average")
    change_threshold: float = Field(
        default=5.0, description="Minimum % change to publish update"
    )
    debounce_seconds: float = Field(
        default=10.0, description="Minimum seconds between state changes"
    )


class MQTTConfig(BaseModel):
    """MQTT broker configuration."""

    host: str = Field(default="localhost", description="MQTT broker hostname")
    port: int = Field(default=1883, description="MQTT broker port")
    username: str = Field(default="", description="MQTT username (optional)")
    password: str = Field(default="", description="MQTT password (optional)")
    client_id: str = Field(default="snowcover", description="MQTT client ID")
    keepalive: int = Field(default=60, description="MQTT keepalive interval")
    discovery_prefix: str = Field(
        default="homeassistant", description="Home Assistant discovery prefix"
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO", description="Log level")
    format: str = Field(default="json", description="Log format (json/text)")


class WebConfig(BaseModel):
    """Web server configuration for live view."""

    enabled: bool = Field(default=False, description="Enable web server for live view")
    host: str = Field(default="0.0.0.0", description="Web server bind address")
    port: int = Field(default=8080, description="Web server port")
    frame_rate: int = Field(default=15, description="MJPEG stream frame rate")


class Settings(BaseSettings):
    """Application settings with YAML and environment variable support."""

    model_config = SettingsConfigDict(
        env_prefix="SNOWCOVER_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    camera: CameraConfig = Field(default_factory=lambda: CameraConfig(rtsp_url=""))
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    smoothing: SmoothingConfig = Field(default_factory=SmoothingConfig)
    mqtt: MQTTConfig = Field(default_factory=MQTTConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    web: WebConfig = Field(default_factory=WebConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Settings":
        """Load settings from a YAML file, with environment variable overrides."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            yaml_config = yaml.safe_load(f) or {}

        return cls(**yaml_config)

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "Settings":
        """Load settings from config file and/or environment variables.

        Priority (highest to lowest):
        1. Environment variables (SNOWCOVER_*)
        2. Config file (if provided)
        3. Default values
        """
        # Check for config file from environment
        if config_path is None:
            config_path = os.environ.get("SNOWCOVER_CONFIG")

        if config_path:
            return cls.from_yaml(config_path)

        # Fall back to pure environment variable configuration
        # This requires RTSP URL to be set
        rtsp_url = os.environ.get("SNOWCOVER_CAMERA__RTSP_URL", "")
        if not rtsp_url:
            raise ValueError(
                "No configuration provided. Either set SNOWCOVER_CONFIG to a YAML file path, "
                "or set SNOWCOVER_CAMERA__RTSP_URL environment variable."
            )

        return cls(camera=CameraConfig(rtsp_url=rtsp_url))
