from __future__ import annotations

import asyncio
import logging
import re
import os
import contextlib
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from music_assistant_models.config_entries import (
    ConfigEntry,
    ConfigEntryType,
    ConfigValueOption,
    ConfigValueType,
)
from music_assistant_models.enums import ContentType, ProviderFeature
from music_assistant_models.media_items import AudioFormat

from music_assistant.models.plugin import PluginProvider, PluginSource

if TYPE_CHECKING:
    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.provider import ProviderManifest


CONF_ALSA_DEVICE = "alsa_device"
CONF_SAMPLE_RATE = "sample_rate"
CONF_CHANNELS = "channels"
CONF_BIT_DEPTH = "bit_depth"
CONF_BUFFER_SIZE = "buffer_size"
CONF_PERIOD_SIZE = "period_size"
CONF_BIT_RATE = "bit_rate"
CONF_QUALITY = "quality"
CONF_PIPE_PATH = "pipe_path"


async def setup(
    mass: MusicAssistant,
    manifest: ProviderManifest,
    config: ProviderConfig,
) -> ProviderInstanceType:
    """Initialization of the provider with the given configuration."""
    prov = ALSACaptureProvider(mass, manifest, config)
    return prov


async def _detect_alsa_devices() -> list[str]:
    """Detection of ALSA devices via arecord -l."""
    devices: list[str] = []
    try:
        proc = await asyncio.create_subprocess_exec(
            "arecord",
            "-l",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode == 0:
            for line in stdout.decode(errors="ignore").splitlines():
                if "card" in line and "device" in line:
                    match = re.search(r"card (\d+).*device (\d+)", line)
                    if match:
                        card_num = match.group(1)
                        device_num = match.group(2)
                        devices.append(f"hw:{card_num},{device_num}")
    except FileNotFoundError:
        logging.getLogger(__name__).warning("arecord not found, using default devices")
    except Exception as err:
        logging.getLogger(__name__).warning("Error detecting ALSA devices: %s", err)
    return devices or ["hw:1,0", "default"]


async def get_config_entries(
    mass: MusicAssistant,
    instance_id: str | None = None,
    action: str | None = None,
    values: dict[str, ConfigValueType] | None = None,
) -> tuple[ConfigEntry, ...]:
    """Provider configuration parameters."""
    devices = await _detect_alsa_devices()
    return (
        ConfigEntry(
            key="instructions",
            type=ConfigEntryType.LABEL,
            label="Instruction",
            description="Select ALSA Capture as the source in the player settings",
            required=False,
        ),
        ConfigEntry(
            key=CONF_ALSA_DEVICE,
            type=ConfigEntryType.STRING,
            label="ALSA device",
            description="Device for capturing audio",
            default_value=devices[0],
            required=True,
            options=[ConfigValueOption(x, x) for x in devices],
        ),
        ConfigEntry(
            key=CONF_PIPE_PATH,
            type=ConfigEntryType.STRING,
            label="Path to FIFO pipe",
            description="Path for the named pipe (e.g. /tmp/alsa_pipe)",
            default_value="/tmp/alsa_capture_pipe",
            required=True,
        ),
        ConfigEntry(
            key=CONF_SAMPLE_RATE,
            type=ConfigEntryType.INTEGER,
            label="Sample rate (Hz)",
            description="Audio sampling rate",
            default_value=44100,
            required=True,
            options=[
                ConfigValueOption(44100, "44100"),
                ConfigValueOption(48000, "48000"),
                ConfigValueOption(88200, "88200"),
                ConfigValueOption(96000, "96000"),
            ],
        ),
        ConfigEntry(
            key=CONF_CHANNELS,
            type=ConfigEntryType.INTEGER,
            label="Number of channels",
            description="Number of audio channels (1-mono, 2-stereo, etc.)",
            default_value=2,
            required=True,
            range=(1, 8),
        ),
        ConfigEntry(
            key=CONF_BIT_DEPTH,
            type=ConfigEntryType.INTEGER,
            label="Bit depth (bits per sample)",
            description="Sampling depth (16, 24, 32)",
            default_value=16,
            required=True,
            options=[
                ConfigValueOption(16, "16"),
                ConfigValueOption(24, "24"),
                ConfigValueOption(32, "32"),
            ],
        ),
        ConfigEntry(
            key=CONF_BUFFER_SIZE,
            type=ConfigEntryType.INTEGER,
            label="Buffer size (frames)",
            description="Smaller values reduce latency but may cause artifacts",
            default_value=128,
            required=True,
            range=(32, 8192),
        ),
        ConfigEntry(
            key=CONF_PERIOD_SIZE,
            type=ConfigEntryType.INTEGER,
            label="Period size (frames)",
            description="Smaller values reduce latency",
            default_value=32,
            required=True,
            range=(16, 4096),
        ),
        ConfigEntry(
            key=CONF_BIT_RATE,
            type=ConfigEntryType.INTEGER,
            label="Bitrate (kbps)",
            description="Bitrate metadata for the UI (PCM is not compressed).",
            default_value=1411,
            required=False,
            options=[
                ConfigValueOption(705, "705 (44.1kHz mono 16-bit)"),
                ConfigValueOption(882, "882 (44.1kHz stereo 16-bit ~)"),
                ConfigValueOption(1411, "1411 (CD quality)"),
                ConfigValueOption(1536, "1536 (48kHz stereo 16-bit)"),
                ConfigValueOption(2304, "2304 (48kHz stereo 24-bit packed)"),
            ],
        ),
        ConfigEntry(
            key=CONF_QUALITY,
            type=ConfigEntryType.STRING,
            label="Quality",
            description="Logical quality level for reference",
            default_value="high",
            required=False,
            options=[
                ConfigValueOption("low", "low"),
                ConfigValueOption("normal", "normal"),
                ConfigValueOption("high", "high"),
                ConfigValueOption("ultra", "ultra"),
            ],
        ),
    )


class ALSACaptureProvider(PluginProvider):
    """Audio capture provider via ALSA using a named pipe."""

    def __init__(
        self,
        mass: MusicAssistant,
        manifest: ProviderManifest,
        config: ProviderConfig,
    ) -> None:
        super().__init__(mass, manifest, config)
        self._device = str(self.config.get_value(CONF_ALSA_DEVICE))
        self._pipe_path = str(self.config.get_value(CONF_PIPE_PATH))
        self._sample_rate = int(self.config.get_value(CONF_SAMPLE_RATE))
        self._channels = int(self.config.get_value(CONF_CHANNELS))
        self._bit_depth = int(self.config.get_value(CONF_BIT_DEPTH))
        self._buffer_size = int(self.config.get_value(CONF_BUFFER_SIZE))
        self._period_size = int(self.config.get_value(CONF_PERIOD_SIZE))
        self._bit_rate_meta = int(self.config.get_value(CONF_BIT_RATE) or 1411)
        self._pipe_proc = None
        
        self._source_details = PluginSource(
            id=self.lookup_key,
            name=f"ALSA Capture: {self._device}",
            passive=False,
            # Completely disable the ability to pause
            can_play_pause=False,
            can_seek=False,
            can_next_previous=False,
            audio_format=AudioFormat(
                content_type=self._content_type_from_bit_depth(),
                codec_type=self._content_type_from_bit_depth(),
                sample_rate=self._sample_rate,
                bit_depth=self._bit_depth,
                channels=self._channels,
                bit_rate=self._bit_rate_meta,
            ),
        )

    def _content_type_from_bit_depth(self) -> ContentType:
        if self._bit_depth == 16:
            return ContentType.PCM_S16LE
        if self._bit_depth == 24:
            return ContentType.PCM_S24LE
        return ContentType.PCM_S32LE

    @property
    def supported_features(self) -> set[ProviderFeature]:
        """Returns the provider's supported features."""
        return {ProviderFeature.AUDIO_SOURCE}

    async def loaded_in_mass(self) -> None:
        """Called after the provider has been fully loaded into Music Assistant."""
        with contextlib.suppress(Exception):
            await self.mass.ui.notify(
                title="ALSA Capture is ready",
                message="Select the source in the player settings",
                type="info",
                duration=8,
            )

    async def _ensure_pipe_exists(self):
        """Ensure that the named pipe exists."""
        try:
            if os.path.exists(self._pipe_path):
                os.unlink(self._pipe_path)
            os.mkfifo(self._pipe_path, 0o666)
            logging.info(f"Created named pipe: {self._pipe_path}")
        except Exception as e:
            logging.error(f"Error creating pipe: {e}")
            raise

    async def get_audio_stream(self, player_id: str) -> AsyncGenerator[bytes, None]:
        """Audio stream generator from the named pipe."""
        await self._ensure_pipe_exists()

        format_str = f"S{self._bit_depth}_LE"
        chunk_size = self._period_size * (self._bit_depth // 8) * self._channels

        # Start arecord to write to the pipe
        self._pipe_proc = await asyncio.create_subprocess_exec(
            "arecord",
            "-D", self._device,
            "-f", format_str,
            "-c", str(self._channels),
            "-r", str(self._sample_rate),
            "--buffer-size", str(self._buffer_size),
            "--period-size", str(self._period_size),
            "-t", "raw",
            "-q",
            self._pipe_path,
            stderr=asyncio.subprocess.PIPE,
        )

        # Read from the pipe
        try:
            with open(self._pipe_path, "rb") as pipe:
                while True:
                    if self._pipe_proc.returncode is not None:
                        break
                        
                    chunk = pipe.read(chunk_size)
                    if not chunk:
                        await asyncio.sleep(0.01)
                        continue
                        
                    yield chunk
                    await asyncio.sleep(0)  # Cooperative multitasking
        except asyncio.CancelledError:
            logging.info("Audio stream was cancelled")
            raise
        except Exception as e:
            logging.error(f"Error in audio stream: {e}")
            raise
        finally:
            await self._cleanup()

    async def _cleanup(self):
        """Resource cleanup."""
        if self._pipe_proc and self._pipe_proc.returncode is None:
            self._pipe_proc.terminate()
            try:
                await asyncio.wait_for(self._pipe_proc.wait(), timeout=2)
            except asyncio.TimeoutError:
                self._pipe_proc.kill()
                await self._pipe_proc.wait()

        try:
            if os.path.exists(self._pipe_path):
                os.unlink(self._pipe_path)
        except Exception as e:
            logging.warning(f"Error removing pipe: {e}")

    async def unload(self, is_removed: bool = False) -> None:
        """Provider unload."""
        await self._cleanup()
        logging.info("ALSA Capture provider unloaded")

    def get_source(self) -> PluginSource:
        """Get information about the audio source."""
        return self._source_details
