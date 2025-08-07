"""
Live-Audio-Input plugin (player-selectable version)
==================================================

Streams realtime PCM from a user-chosen PulseAudio / PipeWire / JACK /
(other FFmpeg device) capture source into *whichever Music-Assistant
player* requests it in the web UI.

No ALSA / arecord binaries are used – capture is done directly via FFmpeg.

Author: you (@yourgithubusername)
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, cast

from music_assistant_models.config_entries import (
    ConfigEntry,
    ConfigEntryType,
)
from music_assistant_models.enums import ContentType, ProviderFeature, StreamType
from music_assistant_models.media_items import AudioFormat
from music_assistant_models.player import PlayerMedia

from music_assistant.helpers.process import AsyncProcess
from music_assistant.models.plugin import PluginProvider, PluginSource

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ConfigValueType, ProviderConfig
    from music_assistant_models.provider import ProviderManifest
    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType

# ────────────────────────────────────────────────────────────────
# Config keys / defaults
# ────────────────────────────────────────────────────────────────
CONF_INPUT_DEVICE  = "input_device"   # FFmpeg device string (pulse:…, jack:…)
CONF_SAMPLE_RATE   = "sample_rate"
CONF_CHANNELS      = "channels"
CONF_FRIENDLY_NAME = "friendly_name"
CONF_BACKEND       = "backend"

DEFAULT_SR       = 48_000
DEFAULT_CHANNELS = 2
DEFAULT_BACKEND  = "pulse"


# ────────────────────────────────────────────────────────────────
# Plugin bootstrap helpers
# ────────────────────────────────────────────────────────────────
async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:  # noqa: D401
    """Instantiate provider."""
    return AudioInputProvider(mass, manifest, config)


async def get_config_entries(  # noqa: D401
    mass: MusicAssistant,
    instance_id: str | None = None,  # noqa: ARG001
    action: str | None = None,       # noqa: ARG001
    values: dict[str, ConfigValueType] | None = None,  # noqa: ARG001
) -> tuple[ConfigEntry, ...]:
    """Wizard: ask only for capture parameters."""
    return (
        ConfigEntry(
            key=CONF_INPUT_DEVICE,
            type=ConfigEntryType.STRING,
            label="Capture device (FFmpeg syntax)",
            description="Pulse example: pulse:bluez_source.XX_XX…   JACK: jack:system:capture_1",
            default_value="default",
            required=True,
        ),
        ConfigEntry(
            key=CONF_BACKEND,
            type=ConfigEntryType.STRING,
            label="FFmpeg avdevice backend",
            default_value=DEFAULT_BACKEND,
            required=True,
            options=[
                {"title": "PulseAudio / PipeWire", "value": "pulse"},
                {"title": "JACK", "value": "jack"},
                {"title": "Other (manual)", "value": "custom"},
            ],
        ),
        ConfigEntry(
            key=CONF_SAMPLE_RATE,
            type=ConfigEntryType.INTEGER,
            label="Sample rate",
            default_value=DEFAULT_SR,
            required=True,
        ),
        ConfigEntry(
            key=CONF_CHANNELS,
            type=ConfigEntryType.INTEGER,
            label="Channels",
            default_value=DEFAULT_CHANNELS,
            required=True,
        ),
        ConfigEntry(
            key=CONF_FRIENDLY_NAME,
            type=ConfigEntryType.STRING,
            label="Display name in UI",
            default_value="Bluetooth Input",
            required=True,
        ),
    )


# ────────────────────────────────────────────────────────────────
# Provider implementation
# ────────────────────────────────────────────────────────────────
class AudioInputProvider(PluginProvider):  # noqa: D101
    #
    # MA calls get_source() once at load-time; players can then pick
    # it in the UI and MA will call get_audio_stream(player_id) each
    # time a *specific* player starts/stops using it.
    #
    def __init__(
        self,
        mass: MusicAssistant,
        manifest: ProviderManifest,
        config: ProviderConfig,
    ) -> None:
        super().__init__(mass, manifest, config)

        # resolve config
        self.device      = cast(str, self.config.get_value(CONF_INPUT_DEVICE))
        self.backend     = cast(str, self.config.get_value(CONF_BACKEND))
        self.sample_rate = cast(int, self.config.get_value(CONF_SAMPLE_RATE))
        self.channels    = cast(int, self.config.get_value(CONF_CHANNELS))
        self.name        = cast(str, self.config.get_value(CONF_FRIENDLY_NAME))

        # static source definition – shared for *any* player
        self._source = PluginSource(
            id=self.instance_id,
            name=self.name,
            passive=False,                     # visible in “Sources” list
            audio_format=AudioFormat(
                codec_type=ContentType.PCM_S16LE,
                content_type=ContentType.PCM_S16LE,
                sample_rate=self.sample_rate,
                bit_depth=16,
                channels=self.channels,
            ),
            metadata=PlayerMedia("Live Audio Input"),
            stream_type=StreamType.CUSTOM,     # we supply bytes via generator
        )

    # ────────── Provider API ──────────
    @property
    def supported_features(self) -> set[ProviderFeature]:  # noqa: D401
        return {ProviderFeature.AUDIO_SOURCE}

    def get_source(self) -> PluginSource:  # noqa: D401
        return self._source

    async def get_audio_stream(  # noqa: D401
        self, player_id: str
    ):  # -> AsyncGenerator[bytes, None]:
        """
        Yield raw PCM to whichever player selected us.

        MA will *automatically* close this generator when the user hits
        stop or picks another source, so we just spawn FFmpeg and pipe
        its stdout until we’re cancelled.
        """
        # Build ffmpeg cmd
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            self.backend,
            "-i",
            self.device,
            "-ac",
            str(self.channels),
            "-ar",
            str(self.sample_rate),
            "-acodec",
            "pcm_s16le",
            "-f",
            "s16le",
            "-fflags",
            "+nobuffer",
            "-flags",
            "+low_delay",
            "-probesize",
            "32",
            "-analyzeduration",
            "0",
            "-",
        ]
        self.logger.info("Starting capture for player %s : %s", player_id, " ".join(cmd))
        proc = AsyncProcess(cmd, stdout=True, stderr=True, name=f"audio-capture[{player_id}]")
        await proc.start()

        try:
            async for chunk in proc.iter_stdout():  # PCM bytes
                yield chunk
        except asyncio.CancelledError:
            # playback stopped
            self.logger.debug("Playback cancelled – stopping FFmpeg.")
            raise
        finally:
            with contextlib.suppress(Exception):
                await proc.close(True)
