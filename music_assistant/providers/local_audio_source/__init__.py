"""
Local Audio Source plugin for Music Assistant

Captures raw PCM from a user-selected ALSA/Pulse (via ALSA) input and forwards it
to a Music Assistant player through an ultra-low-latency CUSTOM stream.

Author: (@Torrax)
"""

from __future__ import annotations

import asyncio
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
    PlayerState,
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

# Debounce/backoff to prevent start/stop thrash during regrouping
PAUSE_DEBOUNCE_S = 0.5
RESUME_DEBOUNCE_S = 0.5
RESTART_BACKOFF_BASE_S = 0.25
RESTART_BACKOFF_MAX_S = 1.5

# ------------------------------------------------------------------
# PROVIDER SET-UP / CONFIG DIALOG
# ------------------------------------------------------------------


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Create plugin instance."""
    return LocalAudioSourceProvider(mass, manifest, config)


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
            default_value="Local Audio Source",
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


class LocalAudioSourceProvider(PluginProvider):
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
        self._capture_lock = asyncio.Lock()
        self._last_state = None
        self._state_since = time.monotonic()
        self._last_start_ts = 0.0
        self._restart_count = 0
        self._active_stream_id: int | None = None

        # Codec management for WAV output
        self._original_codec: str | None = None
        self._codec_changed: bool = False

        # Static plugin-wide audio source definition
        metadata = PlayerMedia("Local Audio Source")
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
        # Monkey patch the streams controller to intercept URL generation for our plugin
        original_get_plugin_source_url = self.mass.streams.get_plugin_source_url

        async def patched_get_plugin_source_url(plugin_source: str, player_id: str) -> str:
            # If this is our plugin, set codec to WAV first
            if plugin_source == self.instance_id:
                self.logger.warning(
                    "🎵 URL GENERATION: Setting codec to WAV for %s before URL generation",
                    self.friendly_name
                )
                await self._save_and_set_wav_codec(player_id)

            # Call the original method
            return await original_get_plugin_source_url(plugin_source, player_id)

        # Replace the method
        self.mass.streams.get_plugin_source_url = patched_get_plugin_source_url
        self.logger.warning(
            "🎵 MONKEY PATCH: Installed codec management for %s",
            self.friendly_name
        )

    async def _save_and_set_wav_codec(self, player_id: str) -> None:
        """Save current codec and set player to WAV format."""
        try:
            # Get current codec setting
            current_codec = await self.mass.config.get_player_config_value(
                player_id, "output_codec"
            )

            self.logger.warning(
                "🎵 CODEC MANAGEMENT: Player %s current codec is '%s'",
                player_id, current_codec
            )

            # Only change if not already WAV
            if current_codec != "wav":
                self._original_codec = current_codec
                self._codec_changed = True

                # Set codec to WAV
                await self.mass.config.save_player_config(
                    player_id=player_id,
                    values={"output_codec": "wav"}
                )

                # Clear any cached config values
                self.mass.config._value_cache.clear()

                # Give the config change time to propagate
                await asyncio.sleep(0.5)

                # Force player update to ensure config is applied
                self.mass.players.update(player_id, force_update=True)

                # Additional wait for player update to complete
                await asyncio.sleep(0.2)

                # Verify the change took effect
                new_codec = await self.mass.config.get_player_config_value(
                    player_id, "output_codec"
                )

                self.logger.warning(
                    "🎵 CODEC CHANGED: Player %s codec changed from '%s' to 'WAV' for %s (verified: %s)",
                    player_id, current_codec, self.friendly_name, new_codec
                )

                if new_codec != "wav":
                    self.logger.error(
                        "🎵 CODEC VERIFICATION FAILED: Expected 'wav' but got '%s' for player %s",
                        new_codec, player_id
                    )
            else:
                self.logger.warning(
                    "🎵 CODEC UNCHANGED: Player %s already using WAV codec for %s",
                    player_id, self.friendly_name
                )

        except Exception as err:
            self.logger.error(
                "🎵 CODEC ERROR: Failed to set WAV codec for player %s: %s",
                player_id, err
            )

    async def _restore_original_codec(self, player_id: str) -> None:
        """Restore the original codec setting."""
        if not self._codec_changed or not self._original_codec:
            self.logger.warning(
                "🎵 CODEC RESTORE: No codec to restore for player %s (changed=%s, original=%s)",
                player_id, self._codec_changed, self._original_codec
            )
            return

        try:
            # Restore original codec
            await self.mass.config.save_player_config(
                player_id=player_id,
                values={"output_codec": self._original_codec}
            )

            self.logger.warning(
                "🎵 CODEC RESTORED: Player %s codec restored from WAV back to '%s' after %s usage",
                player_id, self._original_codec, self.friendly_name
            )

        except Exception as err:
            self.logger.error(
                "🎵 CODEC RESTORE ERROR: Failed to restore codec for player %s: %s",
                player_id, err
            )
        finally:
            # Reset codec management state
            self._original_codec = None
            self._codec_changed = False

    async def _monitor_player_state(self, player_id: str) -> None:
        """Monitor player state with debounce to avoid flapping."""
        self._current_player_id = player_id
        self._last_state = None
        self._state_since = time.monotonic()

        while self._stream_active and not self._stop_called:
            try:
                player = self.mass.players.get(player_id)
                if not player:
                    break

                now = time.monotonic()
                current_state = player.state

                if current_state != self._last_state:
                    # state changed -> reset timer
                    self._last_state = current_state
                    self._state_since = now

                stable_for = now - self._state_since

                # Debounced transitions
                if current_state == PlayerState.PAUSED and stable_for >= PAUSE_DEBOUNCE_S and not self._paused:
                    self.logger.info("Player %s paused (debounced) - stopping arecord", player_id)
                    self._paused = True
                    # stop capture (non-blocking here; loop will perform the stop)
                elif current_state == PlayerState.PLAYING and stable_for >= RESUME_DEBOUNCE_S and self._paused:
                    self.logger.info("Player %s resumed (debounced) - ready to restart arecord", player_id)
                    self._paused = False
                elif current_state == PlayerState.IDLE and stable_for >= PAUSE_DEBOUNCE_S and not self._paused:
                    self.logger.info("Player %s idle (debounced) - stopping arecord", player_id)
                    self._paused = True

                await asyncio.sleep(0.2)

            except Exception as err:
                self.logger.debug("Error monitoring player state: %s", err)
                await asyncio.sleep(1)

    async def unload(self, is_removed: bool = False) -> None:
        """Tear down."""
        self.logger.info("Unloading local audio source provider %s", self.friendly_name)
        self._stop_called = True
        self._stream_active = False

        # Restore codec if we have a current player and changed codec
        if self._current_player_id and self._codec_changed:
            await self._restore_original_codec(self._current_player_id)

        # Stop the capture process first (if any active CUSTOM stream)
        async with self._capture_lock:
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

        self.logger.info("Local audio source provider %s unloaded", self.friendly_name)

    # ---------------- PluginProvider hooks ----------------

    def get_source(self) -> PluginSource:
        """Expose this input as a PlayerSource (CUSTOM stream)."""
        return self._source_details

    async def cmd_pause(self, player_id: str) -> None:
        """Handle pause command for the local audio source stream."""
        self.logger.info("Pausing local audio source stream for %s - stopping arecord but keeping stream alive", self.friendly_name)
        self._paused = True
        async with self._capture_lock:
            if self._capture_proc and not self._capture_proc.closed:
                self.logger.info("Stopping arecord process due to pause for %s", self.friendly_name)
                with suppress(Exception):
                    await self._capture_proc.close(True)
                self._capture_proc = None

    async def cmd_play(self, player_id: str) -> None:
        """Handle play/resume command for the local audio source stream."""
        self.logger.info("Resuming local audio source stream for %s - will restart fresh arecord", self.friendly_name)
        self._paused = False

    async def cmd_stop(self, player_id: str) -> None:
        """Handle stop command for the local audio source stream."""
        self.logger.info("Stopping local audio source stream for %s", self.friendly_name)
        self._paused = False
        self._stream_active = False

        # Restore original codec when stopping
        await self._restore_original_codec(player_id)

        async with self._capture_lock:
            if self._capture_proc and not self._capture_proc.closed:
                self.logger.info("Stopping arecord process due to stop command for %s", self.friendly_name)
                with suppress(Exception):
                    await self._capture_proc.close(True)
                self._capture_proc = None

    async def get_audio_stream(self, player_id: str) -> AsyncGenerator[bytes, None]:
        """Yield raw PCM from arecord directly to MA (low-latency CUSTOM stream)."""
        self._stream_active = True
        self._current_player_id = player_id
        # unique id for this stream instance (prevents cross-talk if a rebuild overlaps)
        self._active_stream_id = (self._active_stream_id or 0) + 1
        my_stream_id = self._active_stream_id

        self.logger.info("Local audio source stream requested for %s by player %s", self.friendly_name, player_id)

        # Ensure codec is WAV (fallback safety)
        current_codec = await self.mass.config.get_player_config_value(player_id, "output_codec")
        if current_codec != "wav":
            self.logger.warning(
                "🎵 FALLBACK CODEC SET: Player %s codec was '%s', setting to WAV",
                player_id, current_codec
            )
            await self._save_and_set_wav_codec(player_id)
        else:
            self.logger.warning(
                "🎵 CODEC VERIFIED: Player %s codec is already 'wav' for %s",
                player_id, self.friendly_name
            )

        # Start player state monitoring
        if not self._monitor_task or self._monitor_task.done():
            self._monitor_task = asyncio.create_task(self._monitor_player_state(player_id))

        # Align chunk size to the requested ALSA period
        bytes_per_sec = self.sample_rate * self.channels * 2  # 16-bit PCM
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
            """Start arecord process with fallback attempts + backoff."""
            nonlocal my_stream_id
            # backoff if we've restarted too quickly
            now = time.monotonic()
            if now - self._last_start_ts < 0.5:
                self._restart_count += 1
            else:
                self._restart_count = 0
            self._last_start_ts = now

            if self._restart_count >= 3:
                backoff = min(RESTART_BACKOFF_BASE_S * (2 ** (self._restart_count - 2)), RESTART_BACKOFF_MAX_S)
                self.logger.warning("Too many rapid arecord restarts (%s). Backing off for %.2fs", self._restart_count, backoff)
                await asyncio.sleep(backoff)

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

        try:
            stream_started = False

            while self._stream_active and not self._stop_called and my_stream_id == self._active_stream_id:
                # Respect paused state (debounced via monitor) — keep stream alive but stop capture
                if self._paused:
                    async with self._capture_lock:
                        if self._capture_proc and not self._capture_proc.closed:
                            self.logger.info("Stopping arecord process for %s (paused)", self.friendly_name)
                            with suppress(Exception):
                                await self._capture_proc.close(True)
                            self._capture_proc = None
                    await asyncio.sleep(period_s)
                    continue

                # Ensure capture exists
                async with self._capture_lock:
                    if not self._capture_proc or self._capture_proc.closed:
                        self.logger.info("Starting fresh arecord process for %s (resumed/playing)", self.friendly_name)
                        self._capture_proc = await start_arecord_process()

                if not self._capture_proc:
                    # Failed to start arecord, wait and retry
                    await asyncio.sleep(period_s)
                    continue

                # Read from arecord
                try:
                    chunk = await asyncio.wait_for(
                        self._capture_proc.read(chunk_size),
                        timeout=period_s * 2
                    )
                    if not chunk:
                        # ended (EOF)
                        self.logger.warning("arecord process ended for %s, will restart on next iteration", self.friendly_name)
                        async with self._capture_lock:
                            self._capture_proc = None
                        await asyncio.sleep(0.05)
                        continue

                    # First chunk -> mark started (but DO NOT flip codec back yet; wait till end)
                    if not stream_started:
                        stream_started = True

                    # Yield audio
                    yield chunk

                except asyncio.TimeoutError:
                    # No data yet; loop
                    continue
                except Exception as err:
                    self.logger.warning("Error reading from arecord for %s: %s", self.friendly_name, err)
                    async with self._capture_lock:
                        if self._capture_proc:
                            with suppress(Exception):
                                await self._capture_proc.close(True)
                            self._capture_proc = None

                await asyncio.sleep(0.001)

        except Exception as err:
            self.logger.error("Error in audio stream for %s: %s", self.friendly_name, err)

        finally:
            # Only the currently-active stream restores codec
            if my_stream_id == self._active_stream_id:
                await self._restore_original_codec(player_id)

            # Clean up monitoring task
            if self._monitor_task and not self._monitor_task.done():
                self._monitor_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._monitor_task
            self._monitor_task = None

            # Clean up capture process
            async with self._capture_lock:
                if self._capture_proc and not self._capture_proc.closed:
                    self.logger.info("Stopping arecord process for %s (stream ended)", self.friendly_name)
                    with suppress(Exception):
                        await self._capture_proc.close(True)
                    self._capture_proc = None

            # mark inactive
            if my_stream_id == self._active_stream_id:
                self._stream_active = False

        self.logger.info("Local audio source stream ended for %s", self.friendly_name)

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
