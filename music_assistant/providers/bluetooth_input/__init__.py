"""
Bluetooth-Input Provider for Music Assistant
(PulseAudio / PipeWire, ultra-low latency, no ALSA required)

Capture live audio from a local Bluetooth receiver (or any Pulse/PipeWire
source) and stream it through Music Assistant with sub-second start-up delay.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from music_assistant.helpers.process import AsyncProcess
from music_assistant.models.music_provider import MusicProvider
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

if TYPE_CHECKING:  # keep runtime deps light
    from music_assistant.mass import MusicAssistant
    from music_assistant_models.provider import ProviderManifest
    from music_assistant.models import ProviderInstanceType

# ────────────────────────────────────────────────────────────────────────────────
# Config constants
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
DEFAULT_BUFFER_MS = 40  # ms

# ────────────────────────────────────────────────────────────────────────────────
# Music Assistant entry points
# ────────────────────────────────────────────────────────────────────────────────
async def setup(
    mass: "MusicAssistant", manifest: "ProviderManifest", config: ProviderConfig
) -> "ProviderInstanceType":
    return BluetoothInputProvider(mass, manifest, config)


async def get_config_entries(  # noqa: PLR0913
    mass: "MusicAssistant",
    instance_id: str | None = None,
    action: str | None = None,
    values: dict[str, ConfigValueType] | None = None,
) -> tuple[ConfigEntry, ...]:
    """Return configuration schema for the provider."""
    return (
        ConfigEntry(
            key=CONF_DISPLAY_NAME,
            type=ConfigEntryType.STRING,
            label="Display Name",
            description="Shown in browse and player bar",
            default_value="Bluetooth Audio Input",
            required=True,
        ),
        ConfigEntry(
            key=CONF_PULSE_SOURCE,
            type=ConfigEntryType.STRING,
            label="PulseAudio / PipeWire source",
            description="e.g. bluez_source.XX.monitor (see `pactl list sources`)",
            default_value="default",
            required=True,
        ),
        ConfigEntry(
            key=CONF_SAMPLE_RATE,
            type=ConfigEntryType.INTEGER,
            label="Sample Rate (Hz)",
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
            label="Buffer (ms)",
            description="Lower → less latency, higher CPU. 20-50 ms is typical.",
            default_value=DEFAULT_BUFFER_MS,
        ),
        ConfigEntry(
            key=CONF_AUTO_START,
            type=ConfigEntryType.BOOLEAN,
            label="Keep Capture Alive",
            description="Start capture immediately when MA boots",
            default_value=True,
        ),
    )

# ────────────────────────────────────────────────────────────────────────────────
# Provider implementation
# ────────────────────────────────────────────────────────────────────────────────
class BluetoothInputProvider(MusicProvider):
    """Real-time Bluetooth audio capture provider."""

    _proc: AsyncProcess | None
    _watchdog: asyncio.Task | None
    _running: bool

    # ─────────── MA lifecycle ───────────
    async def loaded_in_mass(self) -> None:
        await super().loaded_in_mass()
        self._proc = None
        self._watchdog = None
        self._running = False
        if self.config.get_value(CONF_AUTO_START):
            await self._start_capture()

    async def unload(self, is_removed: bool = False) -> None:
        await self._stop_capture()
        await super().unload(is_removed)

    # ─────────── Capabilities ───────────
    @property
    def supported_features(self) -> set[ProviderFeature]:
        return {ProviderFeature.BROWSE, ProviderFeature.LIBRARY_RADIOS}

    @property
    def is_streaming_provider(self) -> bool:
        return False

    # ─────────── Library items ───────────
    async def get_library_radios(self) -> AsyncGenerator[Radio, None]:
        yield self._build_radio()

    async def get_radio(self, prov_radio_id: str) -> Radio:
        if prov_radio_id != BLUETOOTH_INPUT_ID:
            raise MediaNotFoundError(prov_radio_id)
        return self._build_radio()

    # ─────────── Playback ───────────
    async def get_stream_details(
        self, item_id: str, media_type: MediaType  # noqa: ARG002
    ) -> StreamDetails:
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
        if not self._running:
            await self._start_capture()

        assert self._proc and not self._proc.closed
        async for chunk in self._proc.iter_any():
            yield chunk

    # ─────────── Image resolver ───────────
    async def resolve_image(self, path: str) -> str | bytes:
        if path == "icon.svg":
            return (
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
                '<path d="M17.71 7.71 12 2h-1v7.59L6.41 5 5 '
                '6.41 10.59 12 5 17.59 6.41 19 11 14.41V22h1l5.71-5.71L13.41 '
                '12l4.3-4.29M13 5.83 15.17 8 13 10.17V5.83m0 8 '
                'L15.17 16 13 18.17v-4.34Z"/></svg>'
            )
        return path

    # ─────────── Helpers ───────────
    def _build_radio(self) -> Radio:
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

    # ─────────── Capture process ───────────
    async def _start_capture(self) -> None:
        if self._running:
            return

        src = str(self.config.get_value(CONF_PULSE_SOURCE))
        rate = int(self.config.get_value(CONF_SAMPLE_RATE))
        ch = int(self.config.get_value(CONF_CHANNELS))
        buf_ms = int(self.config.get_value(CONF_BUFFER_MS))
        frag = int(rate * ch * (buf_ms / 1000))  # bytes per fragment (approx.)

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            # Input (Pulse/ PipeWire)
            "-f",
            "pulse",
            "-fragment_size",
            str(frag),
            "-i",
            src,
            # Processing / output
            "-ac",
            str(ch),
            "-ar",
            str(rate),
            "-sample_fmt",
            "s16",
            "-acodec",
            "pcm_s16le",
            "-flags",
            "+low_delay",
            "-fflags",
            "+nobuffer",
            "-max_delay",
            "0",
            "-f",
            "s16le",
            "-",  # stdout → MA pipeline
        ]

        self.logger.debug("Starting capture: %s", " ".join(cmd))
        try:
            self._proc = AsyncProcess(cmd, stdout=True, stderr=True)
            await self._proc.start()
        except Exception as err:
            self.logger.error("Unable to start FFmpeg: %s", err)
            raise ProviderUnavailableError(err) from err

        self._running = True
        self._watchdog = asyncio.create_task(self._watchdog_loop())

    async def _stop_capture(self) -> None:
        self._running = False
        if self._watchdog and not self._watchdog.done():
            self._watchdog.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watchdog
        if self._proc and not self._proc.closed:
            await self._proc.close()

    async def _watchdog_loop(self) -> None:
        while self._running:
            await asyncio.sleep(3)
            if (
                self._proc is None
                or self._proc.closed
                or self._proc.returncode is not None
            ):
                self.logger.warning("Capture process died – restarting")
                await self._start_capture()
