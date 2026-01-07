"""MQTT client wrapper with Last Will and Testament support."""

import json
import threading
import time
from typing import Any, Callable

import paho.mqtt.client as mqtt
import structlog

from snowcover.config import MQTTConfig

logger = structlog.get_logger()


class MQTTPublisher:
    """MQTT publisher with connection management and LWT support."""

    def __init__(self, config: MQTTConfig, camera_id: str):
        """Initialize the MQTT publisher.

        Args:
            config: MQTT configuration
            camera_id: Camera identifier for topic naming
        """
        self.config = config
        self.camera_id = camera_id
        self._client: mqtt.Client | None = None
        self._connected = False
        self._lock = threading.Lock()
        self._log = logger.bind(component="mqtt", camera_id=camera_id)

        # Topic prefixes
        self._device_topic = f"snowcover_{camera_id}"
        self._availability_topic = f"{self._device_topic}/availability"

    @property
    def is_connected(self) -> bool:
        """Check if connected to MQTT broker."""
        return self._connected

    @property
    def availability_topic(self) -> str:
        """Get the availability topic for this device."""
        return self._availability_topic

    @property
    def discovery_prefix(self) -> str:
        """Get the Home Assistant discovery prefix."""
        return self.config.discovery_prefix

    @property
    def device_topic(self) -> str:
        """Get the base device topic."""
        return self._device_topic

    def connect(self) -> None:
        """Connect to the MQTT broker."""
        self._log.info(
            "Connecting to MQTT broker",
            host=self.config.host,
            port=self.config.port,
        )

        self._client = mqtt.Client(
            client_id=f"{self.config.client_id}_{self.camera_id}",
            protocol=mqtt.MQTTv5,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        )

        # Set up callbacks
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

        # Callback for HA birth message
        self._on_ha_birth: Callable[[], None] | None = None

        # Authentication
        if self.config.username:
            self._client.username_pw_set(
                self.config.username,
                self.config.password or None,
            )

        # Last Will and Testament - mark device offline if connection lost
        self._client.will_set(
            topic=self._availability_topic,
            payload="offline",
            qos=1,
            retain=True,
        )

        # Connect
        self._client.connect(
            host=self.config.host,
            port=self.config.port,
            keepalive=self.config.keepalive,
        )

        # Start network loop in background thread
        self._client.loop_start()

        # Wait for connection with timeout
        timeout = 10.0
        start = time.time()
        while not self._connected and time.time() - start < timeout:
            time.sleep(0.1)

        if not self._connected:
            raise ConnectionError(
                f"Failed to connect to MQTT broker at {self.config.host}:{self.config.port}"
            )

    def disconnect(self) -> None:
        """Disconnect from the MQTT broker."""
        if self._client:
            # Publish offline status before disconnecting
            self.publish(self._availability_topic, "offline", retain=True)
            time.sleep(0.1)  # Allow message to be sent

            self._client.loop_stop()
            self._client.disconnect()
            self._client = None
            self._connected = False
            self._log.info("Disconnected from MQTT broker")

    def publish(
        self,
        topic: str,
        payload: str | dict[str, Any],
        qos: int = 0,
        retain: bool = False,
    ) -> bool:
        """Publish a message to an MQTT topic.

        Args:
            topic: MQTT topic
            payload: Message payload (string or dict to be JSON-encoded)
            qos: Quality of Service level (0, 1, or 2)
            retain: Whether to retain the message

        Returns:
            True if published successfully, False otherwise
        """
        if not self._client or not self._connected:
            self._log.warning("Cannot publish - not connected", topic=topic)
            return False

        # JSON encode dict payloads
        if isinstance(payload, dict):
            payload = json.dumps(payload)

        with self._lock:
            result = self._client.publish(topic, payload, qos=qos, retain=retain)

        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            self._log.error(
                "Failed to publish message",
                topic=topic,
                error_code=result.rc,
            )
            return False

        return True

    def publish_state(self, entity_type: str, entity_name: str, state: str | int | float) -> bool:
        """Publish state to a Home Assistant entity topic.

        Args:
            entity_type: Entity type (sensor, binary_sensor) - unused but kept for API compat
            entity_name: Entity name suffix
            state: State value

        Returns:
            True if published successfully
        """
        topic = f"{self._device_topic}/{entity_name}/state"
        return self.publish(topic, str(state), retain=True)

    def publish_attributes(
        self,
        entity_type: str,
        entity_name: str,
        attributes: dict[str, Any],
    ) -> bool:
        """Publish JSON attributes to a Home Assistant entity.

        Args:
            entity_type: Entity type (sensor, binary_sensor) - unused but kept for API compat
            entity_name: Entity name suffix
            attributes: Attribute dictionary

        Returns:
            True if published successfully
        """
        topic = f"{self._device_topic}/{entity_name}/attributes"
        return self.publish(topic, attributes, retain=True)

    def publish_availability(self, status: str = "online") -> bool:
        """Publish device availability status.

        Args:
            status: Availability status (online/offline)

        Returns:
            True if published successfully
        """
        return self.publish(self._availability_topic, status, qos=1, retain=True)

    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: mqtt.ConnectFlags,
        reason_code: mqtt.ReasonCode,
        properties: mqtt.Properties | None,
    ) -> None:
        """Handle MQTT connection event."""
        if reason_code == mqtt.CONNACK_ACCEPTED:
            self._connected = True
            self._log.info("Connected to MQTT broker")
            # Subscribe to HA birth message to know when to republish discovery
            birth_topic = f"{self.config.discovery_prefix}/status"
            self._client.subscribe(birth_topic)
            self._log.debug("Subscribed to HA birth topic", topic=birth_topic)
            # Publish online status
            self.publish_availability("online")
        else:
            self._log.error("MQTT connection failed", reason_code=str(reason_code))

    def _on_disconnect(
        self,
        client: mqtt.Client,
        userdata: Any,
        disconnect_flags: mqtt.DisconnectFlags,
        reason_code: mqtt.ReasonCode,
        properties: mqtt.Properties | None,
    ) -> None:
        """Handle MQTT disconnection event."""
        self._connected = False
        self._log.warning("Disconnected from MQTT broker", reason_code=str(reason_code))

    def _on_message(
        self,
        client: mqtt.Client,
        userdata: Any,
        message: mqtt.MQTTMessage,
    ) -> None:
        """Handle incoming MQTT messages."""
        birth_topic = f"{self.config.discovery_prefix}/status"
        if message.topic == birth_topic:
            payload = message.payload.decode("utf-8", errors="ignore")
            if payload == "online":
                self._log.info("Home Assistant birth message received, republishing discovery")
                if self._on_ha_birth:
                    self._on_ha_birth()

    def set_on_ha_birth(self, callback: Callable[[], None]) -> None:
        """Set callback to be called when HA sends birth message.

        Args:
            callback: Function to call (should republish discovery)
        """
        self._on_ha_birth = callback

    def __enter__(self) -> "MQTTPublisher":
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.disconnect()
