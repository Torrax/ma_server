"""
Bluetooth-Input Provider for Music Assistant (pulse/pipewire, low-latency).

Capture live audio from a local Bluetooth sink (or any PulseAudio / PipeWire
source) and stream it through Music Assistant with <1 s start-up delay.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from music_assistant_models.config_entries import (
    ConfigEntry,
    ConfigValueType,
    ProviderConfig,
)
from music_assistant_models.enums import (
    ConfigEntryType,
    ContentType,
    ImageType,
    MediaType,
    ProviderFeature,
    StreamType,
)
from music_assistant_models.errors import MediaNotFoundError, ProviderUnavailableError
from music_assistant_models.media_items import (
    AudioFormat,
    MediaItemImage,
    MediaItemMetadata,
    ProviderMapping,
    Radio,
    UniqueList,
)
from music_assistant_models.streamdetails import StreamDetails

from music_assistant.helpers.process import AsyncProcess
from music_assistant.models.music_provider import MusicProvider

if TYPE_CHECKING:  # Runtime-only imports keep startup fast
    from music_assistant.mass import MusicAssistant
    from music_assistant_models.provider import ProviderManifest
    from music_assistant.models import ProviderInstanceType

# ────────────────────────────────────────────────────────────────────────────────
# Configuration constants
# ────────────────────────────────────────────────────────────────────────────────
BLUETOOTH_INPUT_ID = "bluetooth_input"

CONF_PULSE_SOURCE = "pulse_source"
CONF_SAMPLE_RATE = "sample_rate"
CONF_CHANNELS = "channels"
CONF_BUFFER_MS = "buffer_ms"
CONF_AUTO_START = "auto_start"
CONF_DISPLAY_NAME = "display_name"

DEFAULT_SAMPLE_RATE = 48_000
DEFAULT_CHANNELS = 2
DEFAULT_BUFFER_MS = 40  # ↔ ≈2 k samples @48 kHz – plenty but still snappy

# ────────────────────────────────────────────────────────────────────────────────
# Public entry points
# ────────────────────────────────────────────────────────────────────────────────
async def setup(
    mass: MusicAssistant, manifest: "ProviderManifest", config: ProviderConfig
) -> "ProviderInstanceType":
    return BluetoothInputProvider(mass, manifest, config)


async def get_config_entries(  # noqa: PLR0913
    mass: MusicAssistant,
    instance_id: str | None = None,
    action: str | None = None,
    values: dict[str, ConfigValueType] | None = None,
) -> tuple[ConfigEntry, ...]:
    """Return config UI for this provider."""
    return (
        ConfigEntry(
            key=CONF_DISPLAY_NAME,
            type=ConfigEntryType.STRING,
            label="Display Name",
            description="Shown in player bar and browse view",
            default_value="Bluetooth Audio Input",
            required=True,
        ),
        ConfigEntry(
            key=CONF_PULSE_SOURCE,
            type=ConfigEntryType.STRING,
            label="PulseAudio / PipeWire source",
            description="e.g. bluez_source.XX.monitor  (use pactl list sources)",
            default_value="default",
            required=True,
        ),
        ConfigEntry(
            key=CONF_SAMPLE_RATE,
            type=ConfigEntryType.INTEGER,
            label="Sample rate (Hz)",
            default_value=DEFAULT_SAMPLE_RATE,
        ),
        ConfigEntry(
            key=CONF_CHANNELS,
            type=ConfigEntryType.INTEGER,
            label="Channels",
            default_value=DEFAULT_CHANNELS,
        ),
        ConfigEntry(
            key=CONF_BUFFER_MS,
            type=ConfigEntryType.INTEGER,
            label="FFmpeg buffer (ms)",
            description="Lower = less latency but higher CPU; 20-50 ms is typical",
            default_value=DEFAULT_BUFFER_MS,
        ),
        ConfigEntry(
            key=CONF_AUTO_START,
            type=ConfigEntryType.BOOLEAN,
            label="Keep capture alive",
            description="Start/keep the capture process running as soon as MA starts",
            default_value=False,
        ),
    )

# ────────────────────────────────────────────────────────────────────────────────
# Provider implementation
# ────────────────────────────────────────────────────────────────────────────────
class BluetoothInputProvider(MusicProvider):
    """Real-time Bluetooth audio capture provider (PulseAudio / PipeWire)."""

    _capture_proc: AsyncProcess | None
    _monitor: asyncio.Task | None
    _running: bool

    # ──────────────  MA hooks  ──────────────
    async def loaded_in_mass(self) -> None:
        await super().loaded_in_mass()
        self._capture_proc = None
        self._monitor = None
        self._running = False
        if self.config.get_value(CONF_AUTO_START):
            # Validate configuration before auto-starting
            src = self.config.get_value(CONF_PULSE_SOURCE)
            if src and not src.startswith('<') and 'svg' not in src.lower() and len(src) <= 100:
                try:
                    await self._start_capture()
                except ProviderUnavailableError as err:
                    self.logger.warning("Auto-start failed: %s", err)
            else:
                self.logger.warning("Auto-start disabled due to invalid audio source configuration: %r", src)

    async def unload(self, is_removed: bool = False) -> None:
        await self._stop_capture()
        await super().unload(is_removed)

    # ──────────────  Capabilities  ──────────────
    @property
    def supported_features(self) -> set[ProviderFeature]:
        return {
            ProviderFeature.BROWSE,
            ProviderFeature.LIBRARY_RADIOS,
        }

    @property
    def is_streaming_provider(self) -> bool:  # Library == Catalog here
        return False

    # ──────────────  Library items  ──────────────
    async def get_library_radios(self) -> AsyncGenerator[Radio, None]:
        """Expose the live input as a ‘Radio’ item."""
        yield self._build_radio()

    async def get_radio(self, prov_radio_id: str) -> Radio:
        if prov_radio_id != BLUETOOTH_INPUT_ID:
            raise MediaNotFoundError(prov_radio_id)
        return self._build_radio()

    # ──────────────  Playback  ──────────────
    async def get_stream_details(
        self, item_id: str, media_type: MediaType
    ) -> StreamDetails:  # noqa: D401 – MA signature
        if item_id != BLUETOOTH_INPUT_ID:
            raise MediaNotFoundError(item_id)

        rate = self.config.get_value(CONF_SAMPLE_RATE)
        ch = self.config.get_value(CONF_CHANNELS)

        return StreamDetails(
            provider=self.instance_id,
            item_id=item_id,
            media_type=MediaType.RADIO,
            stream_type=StreamType.CUSTOM,
            can_seek=False,
            allow_seek=False,
            audio_format=AudioFormat(
                content_type=ContentType.PCM_S16LE,
                sample_rate=rate,
                bit_depth=16,
                channels=ch,
            ),
        )

    async def get_audio_stream(
        self, streamdetails: StreamDetails, seek_position: int = 0  # noqa: ARG002
    ) -> AsyncGenerator[bytes, None]:
        """Yield raw PCM directly from the FFmpeg capture process."""
        if not self._running:
            await self._start_capture()

        assert self._capture_proc and not self._capture_proc.closed
        async for chunk in self._capture_proc.iter_any():
            yield chunk

    # ──────────────  Image resolver  ──────────────
    async def resolve_image(self, path: str) -> str | bytes:  # noqa: D401
        if path == "icon.svg":  # served from package dir
            return (
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
                '<path d="M17.71 7.71 12 2h-1v7.59L6.41 5 5 '
                '6.41 10.59 12 5 17.59 6.41 19 11 14.41V22h1l5.71-5.71L13.41 '
                '12l4.3-4.29M13 5.83 15.17 8 13 10.17V5.83m0 8 '
                'L15.17 16 13 18.17v-4.34Z"/></svg>'
            )
        return path

    # ──────────────  Internal helpers  ──────────────
    def _build_radio(self) -> Radio:
        """Return a Radio object reflecting current config."""
        name = self.config.get_value(CONF_DISPLAY_NAME)
        rate = self.config.get_value(CONF_SAMPLE_RATE)
        ch = self.config.get_value(CONF_CHANNELS)

        return Radio(
            item_id=BLUETOOTH_INPUT_ID,
            provider=self.instance_id,
            name=name,
            provider_mappings={
                ProviderMapping(
                    item_id=BLUETOOTH_INPUT_ID,
                    provider_domain=self.domain,
                    provider_instance=self.instance_id,
                    audio_format=AudioFormat(
                        content_type=ContentType.PCM_S16LE,
                        sample_rate=rate,
                        bit_depth=16,
                        channels=ch,
                    ),
                )
            },
            metadata=MediaItemMetadata(
                description="Live audio from local Bluetooth receiver",
                images=UniqueList(
                    [
                        MediaItemImage(
                            type=ImageType.THUMB,
                            path="icon.svg",
                            provider=self.domain,
                            remotely_accessible=False,
                        )
                    ]
                ),
            ),
        )

    # ──────────────  Capture process management  ──────────────
    async def _start_capture(self) -> None:
        if self._running:
            return

        src = self.config.get_value(CONF_PULSE_SOURCE)
        rate = self.config.get_value(CONF_SAMPLE_RATE)
        ch = self.config.get_value(CONF_CHANNELS)
        buf_ms = self.config.get_value(CONF_BUFFER_MS)

        self.logger.info("Attempting to start capture with source: %r", src)

        # Validate the source - make sure it's not SVG content or other invalid input
        if not src or src.startswith('<') or 'svg' in src.lower() or len(src) > 100:
            self.logger.error("Invalid audio source configured (len=%d): %s", len(src) if src else 0, src[:100] if src else "None")
            self._running = False
            raise ProviderUnavailableError(f"Invalid audio source configured. Please check your PulseAudio/PipeWire source setting.")

        try:
            ffmpeg_cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",  # Changed to error to see actual issues
                # Pulse / PipeWire input
                "-f",
                "pulse",
                "-i",
                src,
                # Low-latency flags
                "-flags",
                "+low_delay",
                "-fflags",
                "+nobuffer",
                "-max_delay",
                "0",
                "-use_wallclock_as_timestamps",
                "1",
                "-flush_packets",
                "1",
                # Force PCM output
                "-ac",
                str(ch),
                "-ar",
                str(rate),
                "-sample_fmt",
                "s16",
                "-acodec",
                "pcm_s16le",
                # Reduce internal buffer size
                "-fragment_size",
                str(int(rate * ch * (buf_ms / 1000))),
                # Pipe raw data to stdout
                "-f",
                "s16le",
                "-",
            ]

            self.logger.info("Starting FFmpeg capture: %s", " ".join(ffmpeg_cmd))
            self._capture_proc = AsyncProcess(ffmpeg_cmd, stdout=True, stderr=True)
            await self._capture_proc.start()
            self._running = True

            # Kick off watchdog
            self._monitor = asyncio.create_task(self._watchdog())
            
        except Exception as err:
            self.logger.error("Failed to start capture process: %s", err)
            self._running = False
            raise ProviderUnavailableError(f"Failed to start audio capture: {err}")

    async def _stop_capture(self) -> None:
        self._running = False
        if self._monitor and not self._monitor.done():
            self._monitor.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitor
        if self._capture_proc and not self._capture_proc.closed:
            await self._capture_proc.close()

    async def _watchdog(self) -> None:
        """Restart FFmpeg automatically if it exits unexpectedly."""
        restart_count = 0
        max_restarts = 5
        
        while self._running:
            await asyncio.sleep(3)
            if (
                self._capture_proc is None
                or self._capture_proc.closed
                or self._capture_proc.returncode is not None
            ):
                if restart_count >= max_restarts:
                    self.logger.error("FFmpeg process died too many times (%d), stopping capture", restart_count)
                    self._running = False
                    break
                
                restart_count += 1
                self.logger.warning("Capture process died (attempt %d/%d), restarting in 2 seconds", restart_count, max_restarts)
                await asyncio.sleep(2)
                
                # Restart the process without creating a new watchdog
                if self._running:
                    try:
                        if self._capture_proc and not self._capture_proc.closed:
                            await self._capture_proc.close()
                        
                        src = self.config.get_value(CONF_PULSE_SOURCE)
                        rate = self.config.get_value(CONF_SAMPLE_RATE)
                        ch = self.config.get_value(CONF_CHANNELS)
                        buf_ms = self.config.get_value(CONF_BUFFER_MS)

                        # Validate the source before attempting restart
                        if not src or src.startswith('<') or 'svg' in src.lower():
                            self.logger.error("Invalid audio source configured, stopping restarts: %s", src)
                            self._running = False
                            break

                        ffmpeg_cmd = [
                            "ffmpeg",
                            "-hide_banner",
                            "-loglevel",
                            "warning",
                            # Pulse / PipeWire input
                            "-f",
                            "pulse",
                            "-i",
                            src,
                            # Low-latency flags
                            "-flags",
                            "+low_delay",
                            "-fflags",
                            "+nobuffer",
                            "-max_delay",
                            "0",
                            "-use_wallclock_as_timestamps",
                            "1",
                            "-flush_packets",
                            "1",
                            # Force PCM output
                            "-ac",
                            str(ch),
                            "-ar",
                            str(rate),
                            "-sample_fmt",
                            "s16",
                            "-acodec",
                            "pcm_s16le",
                            # Reduce internal buffer size
                            "-fragment_size",
                            str(int(rate * ch * (buf_ms / 1000))),
                            # Pipe raw data to stdout
                            "-f",
                            "s16le",
                            "-",
                        ]

                        self.logger.debug("Restarting FFmpeg capture: %s", " ".join(ffmpeg_cmd))
                        self._capture_proc = AsyncProcess(ffmpeg_cmd, stdout=True, stderr=True)
                        await self._capture_proc.start()
                        
                    except Exception as err:
                        self.logger.error("Failed to restart capture process: %s", err)
                        continue
            else:
                # Process is running fine, reset restart count
                restart_count = 0
