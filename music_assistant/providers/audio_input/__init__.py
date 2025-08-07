"""
Live-Audio-Input plugin for Music Assistant
==========================================

Captures raw PCM from a user-selected PulseAudio / PipeWire / JACK /
(other FFmpeg device) input and forwards it to a Music Assistant
player through an ultra-low-latency named pipe.

⚠  *No arecord / ALSA binaries are invoked – capture is done
   directly by FFmpeg.*

Author: you (@yourgithubusername)
"""

from __future__ import annotations

import asyncio
import os
from contextlib import suppress
from typing import TYPE_CHECKING, Callable, cast

from music_assistant_models.config_entries import (
    ConfigEntry,
    ConfigEntryType,
    ConfigValueOption,
)
from music_assistant_models.enums import (
    ContentType,
    EventType,
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
    from music_assistant_models.event import MassEvent
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType

# ------------------------------------------------------------------
# CONFIG KEYS
# ------------------------------------------------------------------

CONF_INPUT_DEVICE = "input_device"            # e.g. "pulse:bluez_source.XX_XX"
CONF_SAMPLE_RATE = "sample_rate"             # int (Hz)
CONF_CHANNELS = "channels"                 # 1 or 2
CONF_FRIENDLY_NAME = "friendly_name"           # UI label
CONF_BACKEND = "backend"                 # ffmpeg avdevice (pulse|jack|lavfi...)

DEFAULT_SR = 48000
DEFAULT_CHANNELS = 2
DEFAULT_BACKEND = "pulse"

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
            key=CONF_INPUT_DEVICE,
            type=ConfigEntryType.STRING,
            label="Audio Input Device",
            description="Select an available audio input device",
            options=device_options,
            default_value=device_options[0].value if device_options else "default",
            required=True,
        ),
        ConfigEntry(
            key=CONF_BACKEND,
            type=ConfigEntryType.STRING,
            label="FFmpeg avdevice backend",
            options=[  # extend as needed
                ConfigValueOption("PulseAudio / PipeWire", "pulse"),
                ConfigValueOption("JACK", "jack"),
                ConfigValueOption("Other (manual)", "custom"),
            ],
            default_value=DEFAULT_BACKEND,
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
        ConfigEntry(
            key=CONF_FRIENDLY_NAME,
            type=ConfigEntryType.STRING,
            label="Display name in UI",
            default_value="Bluetooth Input",
            required=True,
        ),
    )


async def _get_available_input_devices() -> list[ConfigValueOption]:
    """Scan for available audio input devices."""
    devices = []
    
    # Try PulseAudio/PipeWire sources first
    try:
        pulse_devices = await _get_pulse_sources()
        devices.extend(pulse_devices)
    except Exception as err:
        # Log but don't fail
        pass
    
    # Try JACK sources
    try:
        jack_devices = await _get_jack_sources()
        devices.extend(jack_devices)
    except Exception as err:
        # Log but don't fail
        pass
    
    # Add fallback options
    if not devices:
        devices = [
            ConfigValueOption("Default Audio Input", "default"),
            ConfigValueOption("Manual Entry (pulse:device_name)", "pulse:"),
            ConfigValueOption("Manual Entry (jack:port_name)", "jack:"),
        ]
    
    return devices


async def _get_pulse_sources() -> list[ConfigValueOption]:
    """Get PulseAudio/PipeWire source devices."""
    devices = []
    
    try:
        # Use pactl to list sources
        returncode, output = await check_output("pactl", "list", "short", "sources")
        if returncode == 0:
            lines = output.decode('utf-8').strip().split('\n')
            for line in lines:
                if line.strip():
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        source_name = parts[1]
                        # Skip monitor sources (they're outputs, not inputs)
                        if '.monitor' not in source_name:
                            # Get friendly name if possible
                            try:
                                returncode2, desc_output = await check_output(
                                    "pactl", "list", "sources"
                                )
                                if returncode2 == 0:
                                    desc_text = desc_output.decode('utf-8')
                                    # Extract description for this source
                                    friendly_name = _extract_pulse_description(desc_text, source_name)
                                    if friendly_name:
                                        display_name = f"{friendly_name} ({source_name})"
                                    else:
                                        display_name = source_name
                                else:
                                    display_name = source_name
                            except Exception:
                                display_name = source_name
                            
                            devices.append(ConfigValueOption(
                                display_name,
                                f"pulse:{source_name}"
                            ))
    except Exception:
        # pactl not available or failed
        pass
    
    return devices


async def _get_jack_sources() -> list[ConfigValueOption]:
    """Get JACK input ports."""
    devices = []
    
    try:
        # Use jack_lsp to list ports
        returncode, output = await check_output("jack_lsp", "-p")
        if returncode == 0:
            lines = output.decode('utf-8').strip().split('\n')
            input_ports = []
            current_port = None
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith('\t'):
                    current_port = line
                elif line.startswith('\t') and 'input' in line and current_port:
                    input_ports.append(current_port)
            
            # Group stereo pairs
            stereo_pairs = []
            mono_ports = []
            
            for port in input_ports:
                if port.endswith('_1') or port.endswith(':1'):
                    base = port[:-2]
                    pair_port = base + '_2' if port.endswith('_1') else base + ':2'
                    if pair_port in input_ports:
                        stereo_pairs.append((port, pair_port))
                    else:
                        mono_ports.append(port)
                elif not any(port.endswith('_2') or port.endswith(':2') for p in input_ports if p.startswith(port[:-2])):
                    mono_ports.append(port)
            
            # Add stereo pairs
            for left, right in stereo_pairs:
                devices.append(ConfigValueOption(
                    f"JACK Stereo: {left.split(':')[0]}",
                    f"jack:{left}|{right}"
                ))
            
            # Add mono ports
            for port in mono_ports:
                devices.append(ConfigValueOption(
                    f"JACK Mono: {port}",
                    f"jack:{port}"
                ))
                
    except Exception:
        # jack_lsp not available or JACK not running
        pass
    
    return devices


def _extract_pulse_description(pactl_output: str, source_name: str) -> str | None:
    """Extract friendly description from pactl list sources output."""
    lines = pactl_output.split('\n')
    in_source = False
    
    for line in lines:
        if f"Name: {source_name}" in line:
            in_source = True
        elif in_source and line.startswith('Source #'):
            in_source = False
        elif in_source and 'Description:' in line:
            desc = line.split('Description:', 1)[1].strip()
            return desc
    
    return None

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
        self.backend: str = cast(str, self.config.get_value(CONF_BACKEND))
        self.sample_rate: int = cast(int, self.config.get_value(CONF_SAMPLE_RATE))
        self.channels: int = cast(int, self.config.get_value(CONF_CHANNELS))
        self.friendly_name: str = cast(str, self.config.get_value(CONF_FRIENDLY_NAME))

        # Runtime helpers
        self.cache_dir = os.path.join(self.mass.cache_path, self.instance_id)
        self.named_pipe = f"/tmp/{self.instance_id}"   # noqa: S108
        self._capture_proc: AsyncProcess | None = None
        self._runner_task: asyncio.Task | None = None          # type: ignore[type-arg]
        self._stop_called = False
        self._capture_started = asyncio.Event()
        self._on_unload_callbacks: list[Callable[..., None]] = []

        # Static plugin-wide audio source definition
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
            metadata=PlayerMedia("Live Audio Input"),
            stream_type=StreamType.NAMED_PIPE,
            path=self.named_pipe,
        )

    # ---------------- Provider API ----------------

    @property
    def supported_features(self) -> set[ProviderFeature]:
        return {ProviderFeature.AUDIO_SOURCE}

    async def handle_async_init(self) -> None:
        """Spin up capture daemon once MA is ready."""
        # Start the capture daemon immediately
        self._start_capture_daemon()

    async def unload(self, is_removed: bool = False) -> None:
        """Tear down."""
        self._stop_called = True
        if self._runner_task and not self._runner_task.done():
            self._runner_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._runner_task
        for cb in self._on_unload_callbacks:
            cb()
        await self._cleanup_pipe()

    # ---------------- PluginProvider hooks ----------------

    def get_source(self) -> PluginSource:
        """Expose the capture device as a PlayerSource."""
        return self._source_details

    # ---------------- Internals ----------------

    def _start_capture_daemon(self) -> None:
        if self._runner_task and not self._runner_task.done():
            return  # already running
        self._runner_task = self.mass.create_task(self._capture_runner())

    def _stop_capture_daemon(self) -> None:
        if self._runner_task and not self._runner_task.done():
            self._runner_task.cancel()

    async def _cleanup_pipe(self) -> None:
        """Remove stale FIFO."""
        await check_output("rm", "-f", self.named_pipe)

    async def _capture_runner(self) -> None:
        """Background task: keep FFmpeg capture alive."""
        self.logger.info("Starting audio capture daemon for %s", self.friendly_name)
        
        # Clean up any existing pipe and create new one
        await self._cleanup_pipe()
        await asyncio.sleep(0.1)
        
        try:
            await check_output("mkfifo", self.named_pipe)
        except Exception as err:
            self.logger.error("Failed to create named pipe %s: %s", self.named_pipe, err)
            return
            
        await asyncio.sleep(0.1)  # ensure pipe exists before ffmpeg starts

        ffmpeg_cmd: list[str] = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "warning",  # Changed from "error" to get more info
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
            self.named_pipe,
        ]

        self.logger.info(
            "Launching FFmpeg capture: %s",
            " ".join(ffmpeg_cmd),
        )

        retry_count = 0
        max_retries = 3

        while not self._stop_called and retry_count < max_retries:
            self._capture_proc = proc = AsyncProcess(
                ffmpeg_cmd, stdout=False, stderr=True, name=f"audio-capture[{self.friendly_name}]"
            )
            
            try:
                await proc.start()

                # Signal that capture has started
                if not self._capture_started.is_set():
                    self._capture_started.set()

                # Collect stderr output to understand what's failing
                stderr_lines = []
                async for line in proc.iter_stderr():
                    stderr_lines.append(line)
                    self.logger.warning("FFmpeg stderr: %s", line)

                # Wait for process to complete and get return code
                return_code = await proc.wait()
                
                if return_code != 0:
                    self.logger.error(
                        "FFmpeg process failed with return code %s. stderr output: %s",
                        return_code,
                        "\n".join(stderr_lines[-10:])  # Last 10 lines
                    )
                    
                    # Check if it's a device issue
                    stderr_text = "\n".join(stderr_lines)
                    if "No such file or directory" in stderr_text or "Device or resource busy" in stderr_text:
                        self.logger.error("Audio device '%s' not available or busy", self.device)
                        break  # Don't retry for device issues
                    
                await proc.close(True)
                
            except Exception as err:
                self.logger.error("Exception in capture process: %s", err)
                with suppress(Exception):
                    await proc.close(True)
            
            if self._stop_called:
                break
                
            retry_count += 1
            if retry_count < max_retries:
                self.logger.warning("Capture process stopped unexpectedly – retrying in 5 s… (attempt %d/%d)", retry_count + 1, max_retries)
                await asyncio.sleep(5)
            else:
                self.logger.error("Max retries reached, stopping capture daemon")
                break

        # Final clean-up
        await self._cleanup_pipe()
