"""
Live-Audio-Input plugin (player-selectable version)
==================================================

Streams realtime PCM from a user-chosen PulseAudio/PipeWire/JACK
(or any FFmpeg avdevice) capture source into *whichever Music-Assistant
player* selects it in the UI.

No ALSA/arecord binaries are used – capture is done directly by FFmpeg.

Author: you (@yourgithubusername)
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, cast

from music_assistant_models.config_entries import (
    ConfigEntry,
    ConfigEntryType,
    ConfigValueOption,
)
from music_assistant_models.enums import ContentType, ProviderFeature, StreamType
from music_assistant_models.media_items import AudioFormat
from music_assistant_models.player import PlayerMedia

from music_assistant.helpers.process import AsyncProcess
from music_assistant.models.plugin import PluginProvider, PluginSource

if TYPE_CHECKING:  # pragma: no cover
    from music_assistant_models.config_entries import ConfigValueType, ProviderConfig
    from music_assistant_models.provider import ProviderManifest
    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType

# ────────────────────────────────────────────────────────────────
# Config keys / defaults
# ────────────────────────────────────────────────────────────────
CONF_INPUT_DEVICE: str = "input_device"   # FFmpeg device string, e.g. pulse:bluez_source.…
CONF_SAMPLE_RATE: str = "sample_rate"
CONF_CHANNELS: str = "channels"
CONF_FRIENDLY_NAME: str = "friendly_name"
CONF_BACKEND: str = "backend"

DEFAULT_SR: int = 48_000
DEFAULT_CHANNELS: int = 2
DEFAULT_BACKEND: str = "pulse"


# ────────────────────────────────────────────────────────────────
# Plugin bootstrap helpers
# ────────────────────────────────────────────────────────────────
async def setup(
    mass: MusicAssistant, manifest: "ProviderManifest", config: "ProviderConfig"
) -> "ProviderInstanceType":
    """Instantiate provider."""
    return AudioInputProvider(mass, manifest, config)


async def get_config_entries(
    mass: MusicAssistant,                       # noqa: ARG001
    instance_id: str | None = None,             # noqa: ARG001
    action: str | None = None,                  # noqa: ARG001
    values: dict[str, "ConfigValueType"] | None = None,  # noqa: ARG001
) -> tuple[ConfigEntry, ...]:
    """Wizard: ask only for capture parameters."""
    return (
        ConfigEntry(
            key=CONF_INPUT_DEVICE,
            type=ConfigEntryType.STRING,
            label="Capture device (FFmpeg syntax)",
            description=(
                "Pulse example: pulse:bluez_source.12_34_56_78_9A_BC  "
                "│  JACK: jack:system:capture_1|system:capture_2"
            ),
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
                ConfigValueOption("PulseAudio / PipeWire", "pulse"),
                ConfigValueOption("JACK", "jack"),
                ConfigValueOption("Other (manual)", "custom"),
            ],
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
            options=[
                ConfigValueOption("Mono", 1),
                ConfigValueOption("Stereo", 2),
            ],
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
class AudioInputProvider(PluginProvider):
    """Realtime audio-capture provider selectable by any player."""

    def __init__(
        self,
        mass: MusicAssistant,
        manifest: "ProviderManifest",
        config: "ProviderConfig",
    ) -> None:
        super().__init__(mass, manifest, config)

        # Resolve config
        self.device: str = cast(str, self.config.get_value(CONF_INPUT_DEVICE))
        self.backend: str = cast(str, self.config.get_value(CONF_BACKEND))
        self.sample_rate: int = cast(int, self.config.get_value(CONF_SAMPLE_RATE))
        self.channels: int = cast(int, self.config.get_value(CONF_CHANNELS))
        self.friendly_name: str = cast(str, self.config.get_value(CONF_FRIENDLY_NAME))

        # Static source definition – shared for any player
        self._source = PluginSource(
            id=self.instance_id,
            name=self.friendly_name,
            passive=False,  # visible in "Sources" list
            audio_format=AudioFormat(
                codec_type=ContentType.PCM_S16LE,
                content_type=ContentType.PCM_S16LE,
                sample_rate=self.sample_rate,
                bit_depth=16,
                channels=self.channels,
            ),
            metadata=PlayerMedia("Live Audio Input"),
            stream_type=StreamType.CUSTOM,  # we supply bytes via generator
        )

    # ───────────── Provider API ─────────────
    @property
    def supported_features(self) -> set[ProviderFeature]:
        return {ProviderFeature.AUDIO_SOURCE}

    def get_source(self) -> PluginSource:
        return self._source

    async def get_audio_stream(self, player_id: str):
        """
        Yield raw PCM to whichever player selected this source.

        MA automatically closes this generator when playback stops;
        we spawn FFmpeg and pipe its stdout until cancelled.
        """
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
        self.logger.info("Starting capture for player %s: %s", player_id, " ".join(cmd))
        proc = AsyncProcess(cmd, stdout=True, stderr=True, name=f"audio-capture[{player_id}]")
        await proc.start()

        try:
            async for chunk in proc.iter_stdout():
                yield chunk
        except asyncio.CancelledError:  # playback stopped
            self.logger.debug("Playback cancelled – stopping FFmpeg for %s", player_id)
            raise
        finally:
            with contextlib.suppress(Exception):
                await proc.close(force_kill=True)
