"""Home Assistant MQTT discovery configuration."""

import json
from dataclasses import dataclass
from typing import Any

import structlog

from snowcover.mqtt.publisher import MQTTPublisher

logger = structlog.get_logger()


@dataclass
class EntityConfig:
    """Configuration for a Home Assistant entity."""

    component: str  # sensor, binary_sensor
    object_id: str  # Unique object ID suffix
    name: str  # Display name
    device_class: str | None = None
    unit_of_measurement: str | None = None
    state_class: str | None = None  # measurement, total, total_increasing
    icon: str | None = None
    entity_category: str | None = None  # config, diagnostic


class HADiscovery:
    """Home Assistant MQTT discovery manager.

    Creates and publishes discovery configurations for all snow detection
    entities, enabling automatic device/entity creation in Home Assistant.
    """

    # Software version
    SW_VERSION = "0.1.0"

    def __init__(self, publisher: MQTTPublisher, camera_id: str):
        """Initialize HA discovery manager.

        Args:
            publisher: MQTT publisher instance
            camera_id: Camera identifier
        """
        self.publisher = publisher
        self.camera_id = camera_id
        self._log = logger.bind(component="ha_discovery", camera_id=camera_id)

        # Device info shared by all entities
        self._device_info = {
            "identifiers": [f"snowcover_{camera_id}"],
            "name": f"Snow Detector - {camera_id}",
            "manufacturer": "SnowCover",
            "model": "ML Snow Detector",
            "sw_version": self.SW_VERSION,
        }

        # Origin info for discovery (helps HA identify the source)
        self._origin_info = {
            "name": "SnowCover",
            "sw_version": self.SW_VERSION,
            "support_url": "https://github.com/snowcover/snowcover",
        }

        # Define all entities
        self._entities = [
            # Binary sensor: Is it snowing?
            EntityConfig(
                component="binary_sensor",
                object_id="snowing",
                name="Snowing",
                icon="mdi:weather-snowy",
            ),
            # Binary sensor: Is ground covered?
            EntityConfig(
                component="binary_sensor",
                object_id="ground_snow",
                name="Ground Snow",
                icon="mdi:snowflake",
            ),
            # Sensor: Particle count
            EntityConfig(
                component="sensor",
                object_id="particle_count",
                name="Snow Particle Count",
                state_class="measurement",
                icon="mdi:dots-hexagon",
            ),
            # Sensor: Snow confidence
            EntityConfig(
                component="sensor",
                object_id="snow_confidence",
                name="Snow Detection Confidence",
                unit_of_measurement="%",
                state_class="measurement",
                icon="mdi:percent-outline",
                entity_category="diagnostic",
            ),
            # Sensor: Snow intensity percentage
            EntityConfig(
                component="sensor",
                object_id="intensity",
                name="Snow Intensity",
                unit_of_measurement="%",
                state_class="measurement",
                icon="mdi:weather-snowy-heavy",
            ),
            # Sensor: Snow intensity category
            EntityConfig(
                component="sensor",
                object_id="intensity_text",
                name="Snow Intensity Level",
                icon="mdi:weather-snowy",
            ),
            # Sensor: Ground coverage percentage
            EntityConfig(
                component="sensor",
                object_id="ground_cover",
                name="Ground Snow Coverage",
                unit_of_measurement="%",
                state_class="measurement",
                icon="mdi:snowflake-variant",
            ),
            # Sensor: Ground confidence
            EntityConfig(
                component="sensor",
                object_id="ground_confidence",
                name="Ground Detection Confidence",
                unit_of_measurement="%",
                state_class="measurement",
                icon="mdi:percent-outline",
                entity_category="diagnostic",
            ),
            # Sensor: Ground brightness
            EntityConfig(
                component="sensor",
                object_id="ground_brightness",
                name="Ground Brightness",
                state_class="measurement",
                icon="mdi:brightness-6",
                entity_category="diagnostic",
            ),
            # Sensor: Detector status
            EntityConfig(
                component="sensor",
                object_id="status",
                name="Detector Status",
                icon="mdi:information-outline",
                entity_category="diagnostic",
            ),
            # Sensor: Camera mode (day/IR)
            EntityConfig(
                component="sensor",
                object_id="camera_mode",
                name="Camera Mode",
                icon="mdi:camera",
                entity_category="diagnostic",
            ),
            # Sensor: CPU usage
            EntityConfig(
                component="sensor",
                object_id="cpu_usage",
                name="CPU Usage",
                unit_of_measurement="%",
                state_class="measurement",
                icon="mdi:cpu-64-bit",
                entity_category="diagnostic",
            ),
        ]

    def _build_discovery_payload(self, entity: EntityConfig) -> dict[str, Any]:
        """Build the MQTT discovery payload for an entity.

        Args:
            entity: Entity configuration

        Returns:
            Discovery payload dictionary
        """
        device_topic = self.publisher.device_topic

        # Base topic for this entity's state (NOT under homeassistant prefix)
        state_topic = f"{device_topic}/{entity.object_id}/state"
        attributes_topic = f"{device_topic}/{entity.object_id}/attributes"

        payload: dict[str, Any] = {
            "name": entity.name,
            "unique_id": f"snowcover_{self.camera_id}_{entity.object_id}",
            "state_topic": state_topic,
            "availability_topic": self.publisher.availability_topic,
            "device": self._device_info,
            "origin": self._origin_info,
            "json_attributes_topic": attributes_topic,
        }

        # Add optional fields
        if entity.device_class:
            payload["device_class"] = entity.device_class

        if entity.unit_of_measurement:
            payload["unit_of_measurement"] = entity.unit_of_measurement

        if entity.state_class:
            payload["state_class"] = entity.state_class

        if entity.icon:
            payload["icon"] = entity.icon

        if entity.entity_category:
            payload["entity_category"] = entity.entity_category

        # Binary sensors use ON/OFF
        if entity.component == "binary_sensor":
            payload["payload_on"] = "ON"
            payload["payload_off"] = "OFF"

        return payload

    def _get_discovery_topic(self, entity: EntityConfig) -> str:
        """Get the discovery topic for an entity.

        Args:
            entity: Entity configuration

        Returns:
            Discovery topic string
        """
        device_topic = self.publisher.device_topic
        discovery_prefix = self.publisher.discovery_prefix

        return f"{discovery_prefix}/{entity.component}/{device_topic}/{entity.object_id}/config"

    def publish_discovery(self) -> bool:
        """Publish discovery configurations for all entities.

        Returns:
            True if all discovery messages were published successfully
        """
        self._log.info("Publishing Home Assistant discovery configurations")

        all_success = True
        for entity in self._entities:
            topic = self._get_discovery_topic(entity)
            payload = self._build_discovery_payload(entity)

            self._log.debug(
                "Publishing discovery",
                topic=topic,
                state_topic=payload.get("state_topic"),
                availability_topic=payload.get("availability_topic"),
            )

            success = self.publisher.publish(
                topic=topic,
                payload=json.dumps(payload),
                qos=1,
                retain=True,
            )

            if success:
                self._log.debug(
                    "Published discovery config",
                    entity=entity.object_id,
                    component=entity.component,
                )
            else:
                self._log.error(
                    "Failed to publish discovery config",
                    entity=entity.object_id,
                )
                all_success = False

        return all_success

    def remove_discovery(self) -> bool:
        """Remove discovery configurations (publish empty payloads).

        This causes Home Assistant to remove the entities.

        Returns:
            True if all removal messages were published successfully
        """
        self._log.info("Removing Home Assistant discovery configurations")

        all_success = True
        for entity in self._entities:
            topic = self._get_discovery_topic(entity)

            # Empty payload removes the entity
            success = self.publisher.publish(
                topic=topic,
                payload="",
                qos=1,
                retain=True,
            )

            if not success:
                all_success = False

        return all_success

    def get_state_topic(self, object_id: str) -> str:
        """Get the state topic for a specific entity.

        Args:
            object_id: Entity object ID (e.g., "snowing", "intensity")

        Returns:
            State topic string
        """
        device_topic = self.publisher.device_topic

        # Find the entity to verify it exists
        entity = next((e for e in self._entities if e.object_id == object_id), None)
        if entity is None:
            raise ValueError(f"Unknown entity: {object_id}")

        return f"{device_topic}/{object_id}/state"

    def publish_state(self, object_id: str, state: str | int | float | bool) -> bool:
        """Publish state for a specific entity.

        Args:
            object_id: Entity object ID
            state: State value

        Returns:
            True if published successfully
        """
        entity = next((e for e in self._entities if e.object_id == object_id), None)
        if entity is None:
            self._log.error("Unknown entity", object_id=object_id)
            return False

        # Convert boolean to ON/OFF for binary sensors
        if entity.component == "binary_sensor":
            state_str = "ON" if state else "OFF"
        else:
            state_str = str(state)

        return self.publisher.publish_state(entity.component, object_id, state_str)

    def publish_attributes(self, object_id: str, attributes: dict[str, Any]) -> bool:
        """Publish attributes for a specific entity.

        Args:
            object_id: Entity object ID
            attributes: Attribute dictionary

        Returns:
            True if published successfully
        """
        entity = next((e for e in self._entities if e.object_id == object_id), None)
        if entity is None:
            self._log.error("Unknown entity", object_id=object_id)
            return False

        return self.publisher.publish_attributes(entity.component, object_id, attributes)
