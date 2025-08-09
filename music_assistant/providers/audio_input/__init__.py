"""
Live-Audio-Input plugin for Music Assistant
==========================================

Captures raw PCM from a user-selected PulseAudio / PipeWire /
(other FFmpeg device) input and forwards it to a Music Assistant
player through an ultra-low-latency stream (CUSTOM provider).

Author: you (@Torrax)
"""

from __future__ import annotations

import asyncio
import os
from contextlib import suppress
from typing import TYPE_CHECKING, Callable, cast
from collections.abc import AsyncGenerator

from music_assistant_models.config_entries import (
    ConfigEntry,
    ConfigEntryType,
    ConfigValueOption,
)
from music_assistant_models.enums import (
    ContentType,
    EventType,
    ImageType,
    ProviderFeature,
    StreamType,
)
from music_assistant_models.media_items import AudioFormat, MediaItemImage
from music_assistant_models.player import PlayerMedia

from music_assistant.constants import CONF_ENTRY_WARN_PREVIEW
from music_assistant.helpers.process import AsyncProcess, check_output
from music_assistant.models.plugin import PluginProvider, PluginSource

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ConfigValueType, ProviderConfig
    from music_assistant_models.event import MassEvent
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType

# ------------------------------------------------------------------
# CONFIG KEYS
# ------------------------------------------------------------------

CONF_INPUT_DEVICE = "input_device"             # e.g. "alsa:hw:1,0"
CONF_SAMPLE_RATE = "sample_rate"               # int (Hz)
CONF_CHANNELS = "channels"                     # 1 or 2
CONF_FRIENDLY_NAME = "friendly_name"           # UI label
CONF_THUMBNAIL_IMAGE = "thumbnail_image"       # Image path/URL

DEFAULT_SR = 48000
DEFAULT_CHANNELS = 2

# ------------------------------------------------------------------
# PROVIDER SET-UP / CONFIG DIALOG
# ------------------------------------------------------------------


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Create plugin instance."""
    return AudioInputProvider(mass, manifest, config)


async def get_config_entries(
    mass: MusicAssistant,
    instance_id: str | None = None,     # noqa: ARG001
    action: str | None = None,          # noqa: ARG001
    values: dict[str, ConfigValueType] | None = None,  # noqa: ARG001
) -> tuple[ConfigEntry, ...]:
    """Config wizard for the plugin."""
    # Get available input devices
    device_options = await _get_available_input_devices()
    
    return (
        CONF_ENTRY_WARN_PREVIEW,
        ConfigEntry(
            key=CONF_FRIENDLY_NAME,
            type=ConfigEntryType.STRING,
            label="Display Name",
            default_value="Bluetooth Input",
            required=True,
        ),
        ConfigEntry(
            key=CONF_THUMBNAIL_IMAGE,
            type=ConfigEntryType.STRING,
            label="Thumbnail Image",
            description="Direct URL to SVG/image file. "
                       "Example: 'https://example.com/icon.svg'",
            default_value="",
            required=False,
        ),
        ConfigEntry(
            key=CONF_INPUT_DEVICE,
            type=ConfigEntryType.STRING,
            label="Audio Input Device",
            description="Select an available audio input device",
            options=device_options,
            default_value=device_options[0].value if device_options else "default",
            required=True,
        ),
        ConfigEntry(
            key=CONF_SAMPLE_RATE,
            type=ConfigEntryType.INTEGER,
            label="Sample rate (Hz)",
            default_value=DEFAULT_SR,
            required=True,
        ),
        ConfigEntry(
            key=CONF_CHANNELS,
            type=ConfigEntryType.INTEGER,
            label="Channels",
            default_value=DEFAULT_CHANNELS,
            required=True,
            options=[ConfigValueOption("Mono", 1), ConfigValueOption("Stereo", 2)],
        ),
    )


async def _get_available_input_devices() -> list[ConfigValueOption]:
    """Scan for available audio input devices."""
    devices = []
    
    # Try ALSA devices first
    try:
        alsa_devices = await _get_alsa_devices()
        devices.extend(alsa_devices)
    except Exception:
        # Log but don't fail
        pass
    
    # Add fallback options
    if not devices:
        devices = [
            ConfigValueOption("Default Audio Input", "default"),
            ConfigValueOption("Manual Entry (alsa:hw:X,Y)", "alsa:"),
        ]
    
    return devices


async def _get_alsa_devices() -> list[ConfigValueOption]:
    """Get ALSA capture devices."""
    devices = []
    
    try:
        # Use arecord to list capture devices
        returncode, output = await check_output("arecord", "-l")
        if returncode == 0:
            lines = output.decode('utf-8').strip().split('\n')
            for line in lines:
                if 'card' in line and 'device' in line:
                    # Parse line like: "card 1: USB [USB Audio], device 0: USB Audio [USB Audio]"
                    if 'card' in line and 'device' in line:
                        try:
                            # Extract card and device numbers
                            card_part = line.split('card ')[1].split(':')[0]
                            device_part = line.split('device ')[1].split(':')[0]
                            
                            # Extract friendly name
                            name_part = line.split(': ')[1] if ': ' in line else f"Card {card_part} Device {device_part}"
                            
                            devices.append(ConfigValueOption(
                                name_part,
                                f"alsa:hw:{card_part},{device_part}"
                            ))
                        except (IndexError, ValueError):
                            # Skip malformed lines
                            continue
    except Exception:
        # arecord not available or failed
        pass
    
    return devices



# ------------------------------------------------------------------
# PROVIDER IMPLEMENTATION
# ------------------------------------------------------------------


class AudioInputProvider(PluginProvider):
    """Realtime audio-capture provider."""

    def __init__(
        self,
        mass: MusicAssistant,
        manifest: ProviderManifest,
        config: ProviderConfig,
    ) -> None:
        super().__init__(mass, manifest, config)

        # Resolve config
        self.device: str = cast(str, self.config.get_value(CONF_INPUT_DEVICE))
        self.sample_rate: int = cast(int, self.config.get_value(CONF_SAMPLE_RATE))
        self.channels: int = cast(int, self.config.get_value(CONF_CHANNELS))
        self.friendly_name: str = cast(str, self.config.get_value(CONF_FRIENDLY_NAME))
        self.thumbnail_image: str = cast(str, self.config.get_value(CONF_THUMBNAIL_IMAGE) or "")
        
        # Parse device string for arecord
        self.ffmpeg_format, self.ffmpeg_device = self._parse_device_string(self.device)

        # Runtime helpers
        self._capture_proc: AsyncProcess | None = None
        self._runner_task: asyncio.Task | None = None          # type: ignore[type-arg]
        self._stop_called = False
        self._capture_started = asyncio.Event()
        self._on_unload_callbacks: list[Callable[..., None]] = []

        # Static plugin-wide audio source definition
        metadata = PlayerMedia("Live Audio Input")
        
        # Add thumbnail image if configured (URLs only)
        if self.thumbnail_image and self.thumbnail_image.startswith(('http://', 'https://')):
            metadata.image_url = self.thumbnail_image
        elif self.thumbnail_image:
            self.logger.warning("Only URLs are supported for thumbnail images. Ignoring: %s", self.thumbnail_image)
        
        self._source_details = PluginSource(
            id=self.instance_id,
            name=self.friendly_name,
            passive=False,                       # can be chosen explicitly by users
            can_play_pause=False,
            can_seek=False,
            can_next_previous=False,
            audio_format=AudioFormat(
                content_type=ContentType.PCM_S16LE,
                codec_type=ContentType.PCM_S16LE,
                sample_rate=self.sample_rate,
                bit_depth=16,
                channels=self.channels,
            ),
            metadata=metadata,
            stream_type=StreamType.CUSTOM,
            path="",  # not used for CUSTOM
        )

    # ---------------- Provider API ----------------

    @property
    def supported_features(self) -> set[ProviderFeature]:
        return {ProviderFeature.AUDIO_SOURCE}

    async def handle_async_init(self) -> None:
        """Called when MA is ready."""
        # No background capture for CUSTOM streams.
        return

    async def unload(self, is_removed: bool = False) -> None:
        """Tear down."""
        self.logger.info("Unloading audio input provider %s", self.friendly_name)
        self._stop_called = True
        
        # Stop the capture process first (if any active CUSTOM stream)
        if self._capture_proc and not self._capture_proc.closed:
            self.logger.info("Terminating capture process for %s", self.friendly_name)
            try:
                await self._capture_proc.close(True)  # Force kill
            except Exception as err:
                self.logger.warning("Error stopping capture process: %s", err)
        
        # Cancel the runner task (not used anymore, but keep for safety)
        if self._runner_task and not self._runner_task.done():
            self._runner_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._runner_task
        
        # Unregister callbacks
        for cb in self._on_unload_callbacks:
            try:
                cb()
            except Exception as err:
                self.logger.warning("Error during callback cleanup: %s", err)
        
        # Force update all players to remove this source from their source lists
        for player in self.mass.players.all():
            # Remove this source from the player's source list
            player.source_list = [
                source for source in player.source_list 
                if source.id != self.instance_id
            ]
            # Update the player to refresh the UI
            self.mass.players.update(player.player_id, force_update=True)
        
        self.logger.info("Audio input provider %s unloaded successfully", self.friendly_name)

    # ---------------- PluginProvider hooks ----------------

    def get_source(self) -> PluginSource:
        """Expose this input as a PlayerSource (CUSTOM stream)."""
        return self._source_details

    async def get_audio_stream(self, player_id: str) -> AsyncGenerator[bytes, None]:
        """Yield raw PCM from arecord directly to MA (low-latency CUSTOM stream).

        We capture S16_LE at the configured sample rate/channels and feed
        small frames (~20 ms) into the MA pipeline. MA will then run a single
        FFmpeg stage for any resampling/output formatting required by the player.
        """
        # assemble arecord command
        # Use small period/buffer times to reduce capture-side latency.
        # (arecord -F/--period-time and -B/--buffer-time take microseconds)
        # If these are unsupported by the device, arecord will error and MA will log it.
        bytes_per_sec = self.sample_rate * self.channels * 2  # 16-bit PCM
        chunk_size = max(1024, bytes_per_sec // 50)  # ~20 ms frames

        cmd: list[str] = [
            "arecord",
            "-D", self.ffmpeg_device,
            "-f", "S16_LE",
            "-c", str(self.channels),
            "-r", str(self.sample_rate),
            "-t", "raw",
            # Lower-latency capture hints (µs). Safe but aggressive defaults.
            "-F", "10000",   # 10 ms period
            "-B", "20000",   # 20 ms buffer
            "-"              # stdout
        ]

        self.logger.info(
            "Starting CUSTOM live-capture for %s (device=%s, sr=%d, ch=%d, chunk=%dB)",
            self.friendly_name, self.ffmpeg_device, self.sample_rate, self.channels, chunk_size
        )

        self._capture_proc = proc = AsyncProcess(
            cmd, stdout=True, stderr=True, name=f"audio-capture[{self.friendly_name}]"
        )
        try:
            await proc.start()
        except Exception as err:
            self.logger.error("Failed to start arecord: %s", err)
            return

        try:
            # stream stdout in small chunks
            async for chunk in proc.iter_chunked(chunk_size):
                if not chunk:
                    break
                yield chunk
        except Exception as err:
            self.logger.error("Error while reading arecord stream: %s", err)
            raise
        finally:
            with suppress(Exception):
                await proc.close(True)
            self._capture_proc = None

    # ---------------- Internals ----------------

    def _parse_device_string(self, device: str) -> tuple[str, str]:
        """Parse device string for arecord."""
        if device.startswith("pulse:"):
            # PulseAudio - fallback to default ALSA device
            return "alsa", "default"
        elif device.startswith("alsa:"):
            return "alsa", device[5:]  # Remove "alsa:" prefix
        elif device == "default":
            return "alsa", "default"
        else:
            # Assume ALSA format/device string
            return "alsa", device
