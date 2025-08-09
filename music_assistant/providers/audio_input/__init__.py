"""
Live-Audio-Input plugin for Music Assistant
===========================================

Captures raw PCM from a user-selected ALSA / PulseAudio / PipeWire input
(via FFmpeg) and exposes it as a low-latency CUSTOM stream (PCM S16LE).

- No encode/decode delay (raw PCM)
- FFmpeg low-latency capture flags + input-side -ac/-ar
- Auto-fallback from hw:N,M -> plughw:N,M
- Preflight device-open test with explicit logging
- Drains FFmpeg stderr (when supported by AsyncProcess) so you see errors
- ~20 ms frame-aligned chunking

Author: you (@Torrax)
"""

from __future__ import annotations

import asyncio
import logging
import time
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
)
from music_assistant_models.media_items import AudioFormat
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
# MODULE LOGGER
# ------------------------------------------------------------------

LOGGER = logging.getLogger(__name__)
LOGGER.debug("live_audio_input module imported")

# ------------------------------------------------------------------
# CONFIG KEYS
# ------------------------------------------------------------------

CONF_INPUT_DEVICE = "input_device"             # e.g. "alsa:hw:1,0" or "pulse:default"
CONF_SAMPLE_RATE = "sample_rate"               # int (Hz)
CONF_CHANNELS = "channels"                     # 1 or 2
CONF_FRIENDLY_NAME = "friendly_name"           # UI label
CONF_THUMBNAIL_IMAGE = "thumbnail_image"       # Image URL (optional)

DEFAULT_SR = 48000
DEFAULT_CHANNELS = 2

# ------------------------------------------------------------------
# PROVIDER SET-UP / CONFIG DIALOG
# ------------------------------------------------------------------

async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Create plugin instance."""
    LOGGER.info("setup() called for live_audio_input; domain=%s", getattr(manifest, "domain", None))
    provider = AudioInputProvider(mass, manifest, config)
    LOGGER.info("setup() created AudioInputProvider instance id=%s", provider.instance_id)
    return provider


async def get_config_entries(
    mass: MusicAssistant,
    instance_id: str | None = None,     # noqa: ARG001
    action: str | None = None,          # noqa: ARG001
    values: dict[str, ConfigValueType] | None = None,  # noqa: ARG001
) -> tuple[ConfigEntry, ...]:
    """Config wizard for the plugin."""
    LOGGER.info("get_config_entries() called; instance_id=%s action=%s", instance_id, action)
    device_options = await _get_available_input_devices()
    LOGGER.info("Device discovery returned %d option(s)", len(device_options))
    for idx, opt in enumerate(device_options):
        name = getattr(opt, "label", None) or getattr(opt, "name", None) or repr(opt)
        val = getattr(opt, "value", None)
        LOGGER.debug("Device option %d: %s -> %s", idx, name, val)

    # Determine safe default
    first_opt_val = getattr(device_options[0], "value", device_options[0]) if device_options else "alsa:default"
    LOGGER.info("Default input device initial value resolved to %s", first_opt_val)

    entries = (
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
            description="Direct URL to an image/SVG (optional). Example: https://example.com/icon.svg",
            default_value="",
            required=False,
        ),
        ConfigEntry(
            key=CONF_INPUT_DEVICE,
            type=ConfigEntryType.STRING,
            label="Audio Input Device",
            description="Pick an input device (ALSA/Pulse/PipeWire via FFmpeg).",
            options=device_options,
            default_value=first_opt_val,
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
    LOGGER.info("get_config_entries() returning %d entries", len(entries))
    return entries


async def _get_available_input_devices() -> list[ConfigValueOption]:
    """Scan for available audio input devices (best effort)."""
    LOGGER.info("_get_available_input_devices() starting")
    devices: list[ConfigValueOption] = []

    # arecord -l
    try:
        LOGGER.debug("Attempting device scan via `arecord -l`")
        returncode, output = await check_output("arecord", "-l")
        LOGGER.debug("arecord returned rc=%s", returncode)
        if returncode == 0:
            lines = output.decode("utf-8", "ignore").splitlines()
            for line in lines:
                if "card " in line and "device " in line:
                    try:
                        card_no = line.split("card ")[1].split(":")[0].strip()
                        dev_no = line.split("device ")[1].split(":")[0].strip()
                        friendly = line.split(": ", 1)[1].strip() if ": " in line else f"Card {card_no} Device {dev_no}"
                        value = f"alsa:hw:{card_no},{dev_no}"
                        label = f"ALSA {friendly}"
                        LOGGER.debug("Detected ALSA device: %s -> %s", label, value)
                        devices.append(ConfigValueOption(label, value))
                    except Exception as err:
                        LOGGER.warning("Failed to parse arecord line: %s; err=%s", line, err)
    except Exception as err:
        LOGGER.info("arecord not available or failed: %s", err)

    # ffmpeg -sources alsa fallback
    if not devices:
        try:
            LOGGER.debug("Attempting device scan via `ffmpeg -sources alsa`")
            returncode, output = await check_output("ffmpeg", "-hide_banner", "-sources", "alsa")
            LOGGER.debug("ffmpeg -sources returned rc=%s", returncode)
            if returncode == 0:
                lines = output.decode("utf-8", "ignore").splitlines()
                seen = set()
                for line in lines:
                    for token in line.replace(",", " ").split():
                        if token.startswith("hw:") and token not in seen:
                            seen.add(token)
                            value = f"alsa:{token}"
                            label = f"ALSA {token}"
                            LOGGER.debug("Detected FFmpeg ALSA token: %s", token)
                            devices.append(ConfigValueOption(label, value))
                if not devices:
                    LOGGER.debug("No explicit hw tokens found; adding ALSA default")
                    devices.append(ConfigValueOption("ALSA default", "alsa:default"))
        except Exception as err:
            LOGGER.info("ffmpeg -sources alsa not available or failed: %s", err)

    # Generic defaults
    if not devices:
        LOGGER.debug("Falling back to generic defaults")
        devices = [
            ConfigValueOption("ALSA default", "alsa:default"),
            ConfigValueOption("PulseAudio default", "pulse:default"),
        ]

    LOGGER.info("_get_available_input_devices() returning %d device(s)", len(devices))
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
        LOGGER.info("AudioInputProvider.__init__ start; instance preparing")
        super().__init__(mass, manifest, config)

        # Resolve config
        self.device: str = cast(str, self.config.get_value(CONF_INPUT_DEVICE))
        self.sample_rate: int = cast(int, self.config.get_value(CONF_SAMPLE_RATE))
        self.channels: int = cast(int, self.config.get_value(CONF_CHANNELS))
        self.friendly_name: str = cast(str, self.config.get_value(CONF_FRIENDLY_NAME))
        self.thumbnail_image: str = cast(str, self.config.get_value(CONF_THUMBNAIL_IMAGE) or "")

        LOGGER.info(
            "Resolved config: friendly_name=%s, device=%s, sample_rate=%d, channels=%d, thumbnail=%s",
            self.friendly_name, self.device, self.sample_rate, self.channels, bool(self.thumbnail_image)
        )

        # Parse device string for FFmpeg (-f <fmt> -i <dev>)
        self._ff_in_format, self._ff_in_device = self._parse_device_string(self.device)
        LOGGER.info("Parsed device: ff_format=%s ff_device=%s", self._ff_in_format, self._ff_in_device)

        # Runtime helpers
        self._capture_proc: AsyncProcess | None = None
        self._runner_task: asyncio.Task | None = None  # type: ignore[type-arg]
        self._stop_called = False
        self._capture_started = asyncio.Event()
        self._on_unload_callbacks: list[callable] = []

        # Metadata
        metadata = PlayerMedia("Live Audio Input")
        if self.thumbnail_image and self.thumbnail_image.startswith(("http://", "https://")):
            with suppress(Exception):
                setattr(metadata, "image_url", self.thumbnail_image)

        self._source_details = PluginSource(
            id=self.instance_id,
            name=self.friendly_name,
            passive=False,
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
            path="",
        )

        LOGGER.info(
            "AudioInputProvider init complete: instance_id=%s, source_name=%s",
            self.instance_id, self.friendly_name
        )

    # ---------------- Provider API ----------------

    @property
    def supported_features(self) -> set[ProviderFeature]:
        LOGGER.debug("supported_features() called")
        return {ProviderFeature.AUDIO_SOURCE}

    async def handle_async_init(self) -> None:
        LOGGER.info("handle_async_init(): provider ready")

    async def unload(self, is_removed: bool = False) -> None:
        LOGGER.info("unload() called; is_removed=%s", is_removed)
        self._stop_called = True
        if self._capture_proc and not self._capture_proc.closed:
            LOGGER.info("Closing capture process for %s", self.friendly_name)
            with suppress(Exception):
                await self._capture_proc.close(True)
        if self._runner_task and not self._runner_task.done():
            LOGGER.info("Cancelling runner task")
            self._runner_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._runner_task
        for idx, cb in enumerate(self._on_unload_callbacks):
            with suppress(Exception):
                LOGGER.debug("Running on_unload callback #%d", idx)
                cb()
        LOGGER.info("unload() complete for %s", self.friendly_name)

    # ---------------- PluginProvider hooks ----------------

    def get_source(self) -> PluginSource:
        LOGGER.info("get_source() called; returning PluginSource id=%s", self._source_details.id)
        return self._source_details

    async def get_audio_stream(self, player_id: str) -> AsyncGenerator[bytes, None]:
        """Yield raw PCM (S16LE) from FFmpeg into MA (CUSTOM stream)."""
        LOGGER.info("get_audio_stream() called for player_id=%s", player_id)

        # Log target player
        try:
            player = self.mass.players.get(player_id)
            LOGGER.info(
                "Target player: id=%s, name=%s, provider=%s",
                getattr(player, "player_id", player_id),
                getattr(player, "display_name", getattr(player, "name", "unknown")),
                getattr(player, "provider", "unknown"),
            )
        except Exception as err:
            LOGGER.warning("Failed to resolve player details for %s: %s", player_id, err)

        # ffmpeg availability
        try:
            rc, ver_out = await check_output("ffmpeg", "-version")
            LOGGER.info("ffmpeg -version rc=%s", rc)
            if rc != 0:
                LOGGER.error("ffmpeg not found or not executable (rc=%s)", rc)
                return
            else:
                ver_line = ver_out.decode("utf-8", "ignore").splitlines()[0:1]
                LOGGER.info("ffmpeg version: %s", " / ".join(ver_line))
        except Exception as err:
            LOGGER.error("ffmpeg not available: %s", err)
            return

        # If ALSA hw: is used, try plughw: fallback if preflight fails
        in_fmt, in_dev = self._ff_in_format, self._ff_in_device
        try_plughw = (in_fmt == "alsa" and in_dev.startswith("hw:"))
        tested_dev = in_dev

        # Preflight open test (captures to null for 0.25s). We don’t need stderr,
        # just OK/non-OK — main run will drain stderr if supported.
        preflight_cmd = [
            "ffmpeg", "-hide_banner", "-nostdin", "-v", "error",
            "-f", in_fmt,
            "-ac", str(self.channels),             # APPLY TO INPUT (placed before -i)
            "-ar", str(self.sample_rate),          # APPLY TO INPUT
            "-i", in_dev,
            "-t", "0.25",
            "-f", "null", "-"                      # throw away
        ]
        LOGGER.info("Preflight: %s", " ".join(preflight_cmd))
        try:
            rc, _ = await check_output(*preflight_cmd)
            LOGGER.info("Preflight result rc=%s (0 means OK)", rc)
            if rc != 0 and try_plughw:
                alt = in_dev.replace("hw:", "plughw:", 1)
                LOGGER.warning("Preflight failed on %s; retrying with %s", in_dev, alt)
                preflight_cmd_alt = preflight_cmd.copy()
                preflight_cmd_alt[preflight_cmd_alt.index(in_dev)] = alt
                rc2, _ = await check_output(*preflight_cmd_alt)
                LOGGER.info("Preflight (plughw) rc=%s", rc2)
                if rc2 == 0:
                    tested_dev = alt
        except Exception as err:
            LOGGER.warning("Preflight exception: %s", err)
            if try_plughw:
                alt = in_dev.replace("hw:", "plughw:", 1)
                LOGGER.warning("Switching to %s due to preflight exception", alt)
                tested_dev = alt

        # Stream params
        bytes_per_sec = self.sample_rate * self.channels * 2  # 16-bit PCM
        frame_bytes = self.channels * 2
        chunk_size = max(1024, (bytes_per_sec // 50) // frame_bytes * frame_bytes)  # ~20 ms
        LOGGER.info(
            "Stream params: sample_rate=%d, channels=%d, bytes_per_sec=%d, frame_bytes=%d, chunk_size=%d",
            self.sample_rate, self.channels, bytes_per_sec, frame_bytes, chunk_size
        )

        # Build FFmpeg command with low-latency flags
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-v", "warning",
            "-f", in_fmt,
            "-ac", str(self.channels),             # INPUT SIDE
            "-ar", str(self.sample_rate),          # INPUT SIDE
            "-thread_queue_size", "512",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-analyzeduration", "0",
            "-probesize", "32",
            "-i", tested_dev,
            "-f", "s16le", "-",                    # raw PCM to stdout
        ]
        LOGGER.info("Launching FFmpeg: %s", " ".join(cmd))

        self._capture_proc = proc = AsyncProcess(
            cmd,
            stdout=True,
            stderr=True,  # we will drain stderr in a background task if supported
            name=f"audio-capture[{self.friendly_name}]"
        )
        stderr_task: asyncio.Task | None = None
        try:
            await proc.start()
            LOGGER.info("FFmpeg process started (pid=%s)", getattr(proc, "pid", None))

            # Start stderr drain if available
            if hasattr(proc, "iter_stderr"):
                async def _drain_stderr() -> None:
                    try:
                        async for line in proc.iter_stderr():  # type: ignore[attr-defined]
                            try:
                                txt = line.decode("utf-8", "ignore").rstrip()
                            except Exception:
                                txt = str(line)
                            if txt:
                                LOGGER.warning("ffmpeg[stderr]: %s", txt)
                    except Exception as e:
                        LOGGER.debug("stderr drain ended: %s", e)
                stderr_task = asyncio.create_task(_drain_stderr())
                LOGGER.debug("Started stderr drain task")
            else:
                LOGGER.debug("AsyncProcess has no iter_stderr(); stderr drain disabled")

        except Exception as err:
            LOGGER.error("Failed to start ffmpeg: %s", err)
            return

        # Streaming loop with byte counters and periodic throughput logs
        total_bytes = 0
        chunk_count = 0
        start_ts = time.monotonic()
        last_log_ts = start_ts

        try:
            async for chunk in proc.iter_chunked(chunk_size):
                if not chunk:
                    LOGGER.warning("Received empty chunk from ffmpeg; breaking")
                    break
                chunk_len = len(chunk)
                total_bytes += chunk_len
                chunk_count += 1

                if chunk_count <= 10:
                    LOGGER.debug("Chunk #%d len=%d", chunk_count, chunk_len)

                now = time.monotonic()
                if now - last_log_ts >= 1.0:
                    elapsed = now - start_ts
                    kb = total_bytes / 1024.0
                    kbps = (kb * 8.0) / elapsed if elapsed > 0 else 0.0
                    LOGGER.info(
                        "Streaming... elapsed=%.2fs chunks=%d bytes=%d (%.1f kbps)",
                        elapsed, chunk_count, total_bytes, kbps
                    )
                    last_log_ts = now

                yield chunk

            LOGGER.info("FFmpeg stdout ended (EOF). chunks=%d bytes=%d", chunk_count, total_bytes)

        except asyncio.CancelledError:
            LOGGER.warning("get_audio_stream() cancelled by caller for player_id=%s", player_id)
            raise
        except Exception as err:
            LOGGER.error("Error while reading ffmpeg stream: %s", err)
            raise
        finally:
            with suppress(Exception):
                await proc.close(True)
            if stderr_task and not stderr_task.done():
                stderr_task.cancel()
                with suppress(asyncio.CancelledError):
                    await stderr_task
            self._capture_proc = None
            elapsed = time.monotonic() - start_ts
            LOGGER.info(
                "Stopped live-capture for %s; elapsed=%.2fs total_chunks=%d total_bytes=%d",
                self.friendly_name, elapsed, chunk_count, total_bytes
            )

    # ---------------- Internals ----------------

    def _parse_device_string(self, device: str) -> tuple[str, str]:
        """Return (ff_input_format, ff_input_device) from a user device string.

        Examples:
          - "alsa:hw:1,0"      -> ("alsa", "hw:1,0")
          - "alsa:default"     -> ("alsa", "default")
          - "pulse:default"    -> ("pulse", "default")
          - "pipewire:default" -> ("pulse", "default")   # via PA shim
          - "default" or ""    -> ("alsa", "default")
        """
        dev = (device or "").strip()
        LOGGER.debug("_parse_device_string() input=%s", dev)
        if dev.startswith("alsa:"):
            result = ("alsa", dev[5:] or "default")
        elif dev.startswith("pulse:"):
            result = ("pulse", dev[6:] or "default")
        elif dev.startswith("pipewire:"):
            result = ("pulse", dev[9:] or "default")
        elif dev == "default" or dev == "":
            result = ("alsa", "default")
        else:
            result = ("alsa", dev)
        LOGGER.debug("_parse_device_string() result=%s", result)
        return result
