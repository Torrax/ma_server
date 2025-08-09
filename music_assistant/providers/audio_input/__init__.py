"""
Live-Audio-Input plugin for Music Assistant
===========================================

Captures raw PCM from a user-selected ALSA / PulseAudio / PipeWire input
(via FFmpeg) and forwards it to a Music Assistant player through a
ultra-low-latency CUSTOM stream.

- Zero extra encode/decode delay (S16LE PCM)
- FFmpeg low-latency capture flags (nobuffer/low_delay, tiny probe)
- ~20 ms chunking, frame-aligned
- Non-passive source so users can pick it directly

Author: you (@Torrax)
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import TYPE_CHECKING, cast
from collections.abc import AsyncGenerator

from music_assistant_models.config_entries import (
    ConfigEntry,
    ConfigEntryType,
    ConfigValueOption,
)
from music_assistant_models.enums import (
    ContentType,
    ProviderFeature,
    StreamType,
    ImageType,
)
from music_assistant_models.media_items import AudioFormat, MediaItemImage
from music_assistant_models.player import PlayerMedia

from music_assistant.constants import CONF_ENTRY_WARN_PREVIEW
from music_assistant.helpers.process import AsyncProcess, check_output
from music_assistant.models.plugin import PluginProvider, PluginSource

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ConfigValueType, ProviderConfig
    from music_assistant_models.provider import ProviderManifest
    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType

# ------------------------------------------------------------------
# CONFIG KEYS
# ------------------------------------------------------------------

CONF_INPUT_DEVICE = "input_device"             # e.g. "alsa:hw:1,0" or "pulse:default"
CONF_SAMPLE_RATE = "sample_rate"               # int (Hz)
CONF_CHANNELS = "channels"                     # 1 or 2
CONF_FRIENDLY_NAME = "friendly_name"           # UI label
CONF_THUMBNAIL_IMAGE = "thumbnail_image"       # Image URL

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
    device_options = await _get_available_input_devices()

    return (
        CONF_ENTRY_WARN_PREVIEW,
        ConfigEntry(
            key=CONF_FRIENDLY_NAME,
            type=ConfigEntryType.STRING,
            label="Display Name",
            default_value="Live Audio Input",
            required=True,
        ),
        ConfigEntry(
            key=CONF_THUMBNAIL_IMAGE,
            type=ConfigEntryType.STRING,
            label="Thumbnail Image (URL)",
            description="Direct URL to an image/SVG. Example: https://example.com/icon.svg",
            default_value="",
            required=False,
        ),
        ConfigEntry(
            key=CONF_INPUT_DEVICE,
            type=ConfigEntryType.STRING,
            label="Audio Input Device",
            description="Pick an input device (ALSA/Pulse/PipeWire via FFmpeg).",
            options=device_options,
            default_value=device_options[0].value if device_options else "alsa:default",
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
    """Scan for available audio input devices (best-effort).

    Prefer ALSA list via `arecord -l`. If unavailable, try `ffmpeg -sources alsa`.
    Fall back to a couple of generic options.
    """
    devices: list[ConfigValueOption] = []

    # Try ALSA capture cards via arecord -l
    try:
        returncode, output = await check_output("arecord", "-l")
        if returncode == 0:
            lines = output.decode("utf-8", "ignore").splitlines()
            for line in lines:
                # Example: "card 1: USB [USB Audio], device 0: USB Audio [USB Audio]"
                if "card " in line and "device " in line:
                    try:
                        card_no = line.split("card ")[1].split(":")[0].strip()
                        dev_no = line.split("device ")[1].split(":")[0].strip()
                        # Friendly name: part after the first ": "
                        if ": " in line:
                            friendly = line.split(": ", 1)[1].strip()
                        else:
                            friendly = f"Card {card_no} Device {dev_no}"
                        devices.append(
                            ConfigValueOption(
                                f"ALSA {friendly}", f"alsa:hw:{card_no},{dev_no}"
                            )
                        )
                    except Exception:
                        # Skip malformed lines
                        continue
    except Exception:
        # arecord not available or failed
        pass

    # Try FFmpeg device enumeration for ALSA as a fallback
    if not devices:
        try:
            returncode, output = await check_output("ffmpeg", "-hide_banner", "-sources", "alsa")
            if returncode == 0:
                lines = output.decode("utf-8", "ignore").splitlines()
                # Heuristic parse: collect "hw:X,Y" tokens if present; otherwise list "default"
                seen = set()
                for line in lines:
                    line = line.strip()
                    # Grab hw:N,M tokens if present
                    for token in line.replace(",", " ").split():
                        if token.startswith("hw:") and token not in seen:
                            seen.add(token)
                            devices.append(
                                ConfigValueOption(f"ALSA {token}", f"alsa:{token}")
                            )
                # Ensure at least a default if FFmpeg is present
                if not devices:
                    devices.append(ConfigValueOption("ALSA default", "alsa:default"))
        except Exception:
            pass

    # Generic fallbacks (Pulse/PipeWire default routes; FFmpeg will resolve)
    if not devices:
        devices = [
            ConfigValueOption("ALSA default", "alsa:default"),
            ConfigValueOption("PulseAudio default", "pulse:default"),
        ]

    return devices


# ------------------------------------------------------------------
# PROVIDER IMPLEMENTATION
# ------------------------------------------------------------------

class AudioInputProvider(PluginProvider):
    """Realtime audio-capture provider (CUSTOM PCM stream)."""

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

        # Parse device string for FFmpeg (-f <fmt> -i <dev>)
        self._ff_in_format, self._ff_in_device = self._parse_device_string(self.device)

        # Runtime helpers
        self._capture_proc: AsyncProcess | None = None
        self._runner_task: asyncio.Task | None = None  # type: ignore[type-arg]
        self._stop_called = False
        self._capture_started = asyncio.Event()
        self._on_unload_callbacks: list[callable] = []

        # Static plugin-wide audio source definition
        metadata = PlayerMedia("Live Audio Input")
        if self.thumbnail_image and self.thumbnail_image.startswith(("http://", "https://")):
            # Prefer proper image object; some UIs also accept metadata.image_url
            metadata.images = [  # type: ignore[attr-defined]
                MediaItemImage(type=ImageType.THUMB, path=self.thumbnail_image)
            ]

        self._source_details = PluginSource(
            id=self.instance_id,
            name=self.friendly_name,
            passive=False,                   # non-passive so users can pick it
            can_play_pause=False,
            can_seek=False,
            can_next_previous=False,
            audio_format=AudioFormat(
                content_type=ContentType.PCM_S16LE,   # raw PCM
                codec_type=ContentType.PCM_S16LE,     # same as content
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
        return

    async def unload(self, is_removed: bool = False) -> None:
        """Tear down."""
        self.logger.info("Unloading audio input provider %s", self.friendly_name)
        self._stop_called = True

        # Stop capture process (if any active CUSTOM stream)
        if self._capture_proc and not self._capture_proc.closed:
            self.logger.info("Terminating capture process for %s", self.friendly_name)
            with suppress(Exception):
                await self._capture_proc.close(True)

        # Cancel background runner if ever used
        if self._runner_task and not self._runner_task.done():
            self._runner_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._runner_task

        # Run any cleanup callbacks
        for cb in self._on_unload_callbacks:
            with suppress(Exception):
                cb()

        self.logger.info("Audio input provider %s unloaded", self.friendly_name)

    # ---------------- PluginProvider hooks ----------------

    def get_source(self) -> PluginSource:
        """Expose this input as a PlayerSource (CUSTOM stream)."""
        return self._source_details

    async def get_audio_stream(self, player_id: str) -> AsyncGenerator[bytes, None]:
        """Yield raw PCM (S16LE) from FFmpeg into MA (CUSTOM stream).

        We capture at the configured sample rate/channels and feed
        small, frame-aligned chunks (~20 ms) into MA's pipeline.
        """
        bytes_per_sec = self.sample_rate * self.channels * 2  # 16-bit PCM
        frame_bytes = self.channels * 2
        # Align chunk to full frames; default ~20 ms
        chunk_size = max(1024, (bytes_per_sec // 50) // frame_bytes * frame_bytes)

        # Build FFmpeg command with low-latency flags
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-f", self._ff_in_format,
            "-thread_queue_size", "512",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-analyzeduration", "0",
            "-probesize", "32",
            "-i", self._ff_in_device,
            "-ac", str(self.channels),
            "-ar", str(self.sample_rate),
            "-f", "s16le", "-",  # raw PCM to stdout
        ]

        self.logger.info(
            "Starting CUSTOM live-capture for %s (fmt=%s, dev=%s, %d Hz, ch=%d, chunk=%dB)",
            self.friendly_name, self._ff_in_format, self._ff_in_device,
            self.sample_rate, self.channels, chunk_size
        )

        self._capture_proc = proc = AsyncProcess(
            cmd, stdout=True, stderr=True, name=f"audio-capture[{self.friendly_name}]"
        )
        try:
            await proc.start()
        except Exception as err:
            self.logger.error("Failed to start ffmpeg: %s", err)
            return

        try:
            async for chunk in proc.iter_chunked(chunk_size):
                if not chunk:
                    break
                # Yield raw PCM bytes directly to MA's stream server
                yield chunk
        except Exception as err:
            self.logger.error("Error while reading ffmpeg stream: %s", err)
            raise
        finally:
            with suppress(Exception):
                await proc.close(True)
            self._capture_proc = None

    # ---------------- Internals ----------------

    def _parse_device_string(self, device: str) -> tuple[str, str]:
        """Return (ff_input_format, ff_input_device) from a user device string.

        Supported examples:
          - "alsa:hw:1,0"      -> ("alsa", "hw:1,0")
          - "alsa:default"     -> ("alsa", "default")
          - "pulse:default"    -> ("pulse", "default")
          - "default"          -> ("alsa", "default")
        """
        dev = device.strip()
        if dev.startswith("alsa:"):
            return "alsa", dev[5:] or "default"
        if dev.startswith("pulse:"):
            return "pulse", dev[6:] or "default"
        if dev.startswith("pipewire:"):
            # FFmpeg uses "pulse" for PipeWire via the PulseAudio shim on most systems.
            # If native pipewire device is needed, adjust here (e.g., "pipewire")
            return "pulse", "default"
        if dev == "default":
            return "alsa", "default"
        # Fallback: assume ALSA device token (e.g., "hw:1,0")
        return "alsa", dev
