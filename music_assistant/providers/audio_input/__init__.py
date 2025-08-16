"""
Live-Audio-Input plugin for Music Assistant
==========================================

Captures raw PCM from a user-selected ALSA/Pulse (via ALSA) input and forwards it
to a Music Assistant player through an ultra-low-latency CUSTOM stream.

Author: (@Torrax)
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
# CONFIG KEYS
# ------------------------------------------------------------------

CONF_INPUT_DEVICE = "input_device"             # e.g. "alsa:hw:1,0" or "default"
CONF_SAMPLE_RATE = "sample_rate"               # int (Hz)
CONF_CHANNELS = "channels"                     # 1 or 2
CONF_FRIENDLY_NAME = "friendly_name"           # UI label
CONF_THUMBNAIL_IMAGE = "thumbnail_image"       # URL only (for now)
CONF_PERIOD_US = "period_us"                   # ALSA period in microseconds
CONF_BUFFER_US = "buffer_us"                   # ALSA buffer in microseconds

DEFAULT_SR = 44100
DEFAULT_CHANNELS = 2
DEFAULT_PERIOD_US = 10000    # 10 ms
DEFAULT_BUFFER_US = 20000    # 20 ms

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
            key="display_section",
            type=ConfigEntryType.LABEL,
            label="------------ DISPLAY ------------",
        ),
        ConfigEntry(
            key=CONF_FRIENDLY_NAME,
            type=ConfigEntryType.STRING,
            label="Display Name",
            default_value="Live Line-In",
            required=True,
        ),
        ConfigEntry(
            key=CONF_THUMBNAIL_IMAGE,
            type=ConfigEntryType.STRING,
            label="Thumbnail image",
            description="Direct URL to an SVG/PNG/JPG, e.g. https://example.com/icon.svg",
            default_value="",
            required=False,
        ),
        ConfigEntry(
            key="audio_section",
            type=ConfigEntryType.LABEL,
            label="------------- AUDIO -------------",
        ),
        ConfigEntry(
            key=CONF_INPUT_DEVICE,
            type=ConfigEntryType.STRING,
            label="Audio Input Device",
            description="Select an ALSA capture device (arecord -l).",
            options=device_options,
            default_value=(device_options[0].value if device_options else "default"),
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
            key=CONF_PERIOD_US,
            type=ConfigEntryType.INTEGER,
            label="Period (µs)",
            description="ALSA period time for arecord (-F). Lower = lower latency, higher risk of XRUNs.",
            default_value=DEFAULT_PERIOD_US,
            required=True,
        ),
        ConfigEntry(
            key=CONF_BUFFER_US,
            type=ConfigEntryType.INTEGER,
            label="Buffer (µs)",
            description="ALSA buffer time for arecord (-B). Should be a small multiple of period.",
            default_value=DEFAULT_BUFFER_US,
            required=True,
        ),
    )


async def _get_available_input_devices() -> list[ConfigValueOption]:
    """Scan for available ALSA capture devices using arecord -l."""
    devices: list[ConfigValueOption] = []
    try:
        # check_output returns (returncode, stdout_bytes)
        returncode, output = await check_output("arecord", "-l")
        if returncode == 0:
            lines = output.decode("utf-8", "ignore").strip().splitlines()
            # Sample lines:
            # "card 1: USB [USB Audio], device 0: USB Audio [USB Audio]"
            for line in lines:
                if "card " in line and "device " in line:
                    try:
                        card = line.split("card ")[1].split(":")[0].strip()
                        device = line.split("device ")[1].split(":")[0].strip()
                        # Friendly name portion after the first colon+space
                        name = line.split(": ", 1)[1].strip() if ": " in line else f"Card {card} Device {device}"
                        devices.append(
                            ConfigValueOption(name, f"alsa:hw:{card},{device}")
                        )
                    except Exception:
                        continue
    except Exception:
        # arecord not available or failed - will fall back
        pass

    if not devices:
        devices = [
            ConfigValueOption("Default Audio Input (ALSA)", "default"),
            ConfigValueOption("Manual Entry (alsa:hw:X,Y)", "alsa:"),
        ]
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
        self.period_us: int = cast(int, self.config.get_value(CONF_PERIOD_US))
        self.buffer_us: int = cast(int, self.config.get_value(CONF_BUFFER_US))
        self.friendly_name: str = cast(str, self.config.get_value(CONF_FRIENDLY_NAME))
        self.thumbnail_image: str = cast(str, self.config.get_value(CONF_THUMBNAIL_IMAGE) or "")

        # Parse device string for ALSA (arecord)
        self.alsa_device = self._parse_device_string(self.device)

        # Runtime helpers
        self._capture_proc: AsyncProcess | None = None
        self._runner_task: asyncio.Task | None = None          # type: ignore[type-arg]
        self._stop_called = False
        self._paused = False
        self._stream_active = False
        self._current_player_id: str | None = None
        self._monitor_task: asyncio.Task | None = None          # type: ignore[type-arg]

        # Static plugin-wide audio source definition
        metadata = PlayerMedia("Live Audio Input")
        if self.thumbnail_image and self.thumbnail_image.startswith(("http://", "https://")):
            metadata.image_url = self.thumbnail_image
        elif self.thumbnail_image:
            self.logger.warning(
                "Only URLs are supported for thumbnail images. Ignoring: %s",
                self.thumbnail_image,
            )

        self._source_details = PluginSource(
            id=self.instance_id,
            name=self.friendly_name,
            passive=False,                       # user-selectable source
            can_play_pause=True,
            can_seek=False,
            can_next_previous=False,
            audio_format=AudioFormat(
                content_type=ContentType.PCM_S16LE,  # raw PCM
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

    async def _monitor_player_state(self, player_id: str) -> None:
        """Monitor player state to detect pause/play/stop commands."""
        from music_assistant_models.enums import PlayerState
        
        previous_state = None
        self._current_player_id = player_id
        
        while self._stream_active and not self._stop_called:
            try:
                # Get the current player state
                player = self.mass.players.get(player_id)
                if not player:
                    break
                    
                current_state = player.state
                
                # Check if player state changed
                if previous_state != current_state:
                    if current_state == PlayerState.PAUSED and not self._paused:
                        self.logger.info("Player %s paused - will stop arecord process for %s", player_id, self.friendly_name)
                        self._paused = True
                    elif current_state == PlayerState.PLAYING and self._paused:
                        self.logger.info("Player %s resumed - will restart arecord process for %s", player_id, self.friendly_name)
                        self._paused = False
                    elif current_state == PlayerState.IDLE:
                        self.logger.info("Player %s stopped - will stop arecord process for %s", player_id, self.friendly_name)
                        self._paused = True
                
                previous_state = current_state
                await asyncio.sleep(0.2)  # Check every 200ms
                
            except Exception as err:
                self.logger.debug("Error monitoring player state: %s", err)
                await asyncio.sleep(1)  # Wait longer on error

    async def unload(self, is_removed: bool = False) -> None:
        """Tear down."""
        self.logger.info("Unloading audio input provider %s", self.friendly_name)
        self._stop_called = True
        self._stream_active = False

        # Stop the capture process first (if any active CUSTOM stream)
        if self._capture_proc and not self._capture_proc.closed:
            self.logger.info("Terminating capture process for %s", self.friendly_name)
            with suppress(Exception):
                await self._capture_proc.close(True)
            self._capture_proc = None

        # Cancel the monitor task
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._monitor_task
        self._monitor_task = None

        # Cancel the runner task (defensive)
        if self._runner_task and not self._runner_task.done():
            self._runner_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._runner_task

        # Optionally force players to refresh their source lists
        for player in self.mass.players.all():
            try:
                player.source_list = [
                    source for source in player.source_list if source.id != self.instance_id
                ]
                self.mass.players.update(player.player_id, force_update=True)
            except Exception:
                continue

        self.logger.info("Audio input provider %s unloaded", self.friendly_name)

    # ---------------- PluginProvider hooks ----------------

    def get_source(self) -> PluginSource:
        """Expose this input as a PlayerSource (CUSTOM stream)."""
        return self._source_details

    async def cmd_pause(self, player_id: str) -> None:
        """Handle pause command for the audio input stream."""
        self.logger.info("Pausing audio input stream for %s - stopping arecord but keeping stream alive", self.friendly_name)
        self._paused = True
        
        # Stop the current capture process
        if self._capture_proc and not self._capture_proc.closed:
            self.logger.info("Stopping arecord process due to pause for %s", self.friendly_name)
            with suppress(Exception):
                await self._capture_proc.close(True)
            self._capture_proc = None

    async def cmd_play(self, player_id: str) -> None:
        """Handle play/resume command for the audio input stream."""
        self.logger.info("Resuming audio input stream for %s - will restart fresh arecord", self.friendly_name)
        self._paused = False
        # The arecord process will be restarted fresh in the stream loop

    async def cmd_stop(self, player_id: str) -> None:
        """Handle stop command for the audio input stream."""
        self.logger.info("Stopping audio input stream for %s", self.friendly_name)
        self._paused = False
        self._stream_active = False
        
        # Stop the current capture process
        if self._capture_proc and not self._capture_proc.closed:
            self.logger.info("Stopping arecord process due to stop command for %s", self.friendly_name)
            with suppress(Exception):
                await self._capture_proc.close(True)
            self._capture_proc = None

    async def get_audio_stream(self, player_id: str) -> AsyncGenerator[bytes, None]:
        """Yield raw PCM from arecord directly to MA (low-latency CUSTOM stream)."""
        self._stream_active = True
        self._current_player_id = player_id
        self.logger.info("Audio input stream requested for %s by player %s", self.friendly_name, player_id)

        # Start player state monitoring
        if not self._monitor_task or self._monitor_task.done():
            self._monitor_task = asyncio.create_task(self._monitor_player_state(player_id))

        # Align chunk size to the requested ALSA period
        bytes_per_sec = self.sample_rate * self.channels * 2  # 16-bit PCM
        # period in seconds
        period_s = max(1, self.period_us) / 1_000_000
        chunk_size = max(256, int(bytes_per_sec * period_s))

        # Primary (aggressive) arecord command
        base_cmd: list[str] = [
            "arecord",
            "-D", self.alsa_device,
            "-f", "S16_LE",
            "-c", str(self.channels),
            "-r", str(self.sample_rate),
            "-t", "raw",
            "-M",              # use mmap for lower latency
            "-q",              # quiet (only errors)
            "-F", str(self.period_us),
            "-B", str(self.buffer_us),
            "-"                # stdout
        ]

        # Fallback settings if the device rejects aggressive period/buffer
        fallbacks = [
            {"F": self.period_us * 2, "B": self.buffer_us * 3},     # ~20 ms / 60 ms
            {"F": 40000, "B": 120000},                               # 40 ms / 120 ms
        ]

        # Try primary, then fallbacks
        attempt_cmds: list[list[str]] = [base_cmd]
        for fb in fallbacks:
            attempt_cmds.append([
                "arecord",
                "-D", self.alsa_device,
                "-f", "S16_LE",
                "-c", str(self.channels),
                "-r", str(self.sample_rate),
                "-t", "raw",
                "-M",
                "-q",
                "-F", str(fb["F"]),
                "-B", str(fb["B"]),
                "-"
            ])

        async def start_arecord_process() -> AsyncProcess | None:
            """Start arecord process with fallback attempts."""
            last_err: Exception | None = None
            
            for idx, cmd in enumerate(attempt_cmds, start=1):
                F_val = cmd[cmd.index("-F")+1]
                B_val = cmd[cmd.index("-B")+1]
                self.logger.info(
                    "Starting CUSTOM live-capture for %s (device=%s, sr=%d, ch=%d, period=%sµs, buffer=%sµs, chunk=%dB) [attempt %d/%d]",
                    self.friendly_name, self.alsa_device, self.sample_rate, self.channels,
                    F_val, B_val, chunk_size, idx, len(attempt_cmds)
                )

                proc = AsyncProcess(cmd, stdout=True, stderr=True, name=f"audio-capture[{self.friendly_name}]")
                try:
                    await proc.start()
                    return proc
                except Exception as err:
                    last_err = err
                    self.logger.warning("arecord failed to start (attempt %d): %s", idx, err)
                    continue
            
            self.logger.error("All arecord attempts failed for %s: %s", self.friendly_name, last_err)
            return None

        # Start initial arecord process only if not paused
        if not self._paused:
            self._capture_proc = await start_arecord_process()
        
        try:
            # Main streaming loop - keep stream alive but control audio output
            while self._stream_active and not self._stop_called:
                if self._paused:
                    # Paused state - stop arecord if running and just wait without yielding
                    if self._capture_proc and not self._capture_proc.closed:
                        self.logger.info("Stopping arecord process for %s (paused)", self.friendly_name)
                        with suppress(Exception):
                            await self._capture_proc.close(True)
                        self._capture_proc = None
                    
                    # Just wait during pause - don't yield anything, keep stream alive
                    await asyncio.sleep(period_s)
                    continue
                
                # Playing state - ensure arecord is running
                if not self._capture_proc or self._capture_proc.closed:
                    self.logger.info("Starting fresh arecord process for %s (resumed)", self.friendly_name)
                    self._capture_proc = await start_arecord_process()
                    
                    if not self._capture_proc:
                        # Failed to start arecord, wait and retry
                        await asyncio.sleep(period_s)
                        continue
                
                # Read from arecord process
                try:
                    chunk = await asyncio.wait_for(
                        self._capture_proc.read(chunk_size), 
                        timeout=period_s * 2
                    )
                    
                    if chunk:
                        yield chunk
                    else:
                        # arecord ended, mark for restart
                        self.logger.warning("arecord process ended for %s, will restart on next iteration", self.friendly_name)
                        self._capture_proc = None
                        
                except asyncio.TimeoutError:
                    # No data available, continue loop
                    continue
                except Exception as err:
                    self.logger.warning("Error reading from arecord for %s: %s", self.friendly_name, err)
                    # Mark process for restart
                    if self._capture_proc:
                        with suppress(Exception):
                            await self._capture_proc.close(True)
                        self._capture_proc = None
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.001)

        except Exception as err:
            self.logger.error("Error in audio stream for %s: %s", self.friendly_name, err)

        finally:
            # Clean up monitoring task
            if self._monitor_task and not self._monitor_task.done():
                self._monitor_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._monitor_task
            self._monitor_task = None
            
            # Clean up capture process
            if self._capture_proc and not self._capture_proc.closed:
                self.logger.info("Stopping arecord process for %s (stream ended)", self.friendly_name)
                with suppress(Exception):
                    await self._capture_proc.close(True)
                self._capture_proc = None

        self.logger.info("Audio stream ended for %s", self.friendly_name)

    # ---------------- Internals ----------------

    def _parse_device_string(self, device: str) -> str:
        """Normalize device string for arecord."""
        if device.startswith("alsa:"):
            return device[5:] or "default"
        if device.startswith("pulse:"):
            # Pulse via ALSA plugin (if present). arecord -D pulse
            return "pulse"
        if device in ("default", ""):
            return "default"
        # Assume it's already a valid ALSA device name
        return device
