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
    ImageType,
    ProviderFeature,
    StreamType,
)
from music_assistant_models.media_items import AudioFormat, MediaItemImage
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

CONF_INPUT_DEVICE = "input_device"            # e.g. "alsa:hw:1,0"
CONF_SAMPLE_RATE = "sample_rate"             # int (Hz)
CONF_CHANNELS = "channels"                 # 1 or 2
CONF_FRIENDLY_NAME = "friendly_name"           # UI label
CONF_THUMBNAIL_IMAGE = "thumbnail_image"       # Image path/URL

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
    # Get available input devices
    device_options = await _get_available_input_devices()
    
    return (
        CONF_ENTRY_WARN_PREVIEW,
        ConfigEntry(
            key=CONF_FRIENDLY_NAME,
            type=ConfigEntryType.STRING,
            label="Display Name",
            default_value="Bluetooth Input",
            required=True,
        ),
        ConfigEntry(
            key=CONF_THUMBNAIL_IMAGE,
            type=ConfigEntryType.STRING,
            label="Thumbnail Image",
            description="Path to image file (relative to provider directory) or direct URL to SVG/image file. "
                       "Examples: 'images/bluetooth.svg' or 'https://example.com/icon.svg'",
            default_value="",
            required=False,
        ),
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
    """Scan for available audio input devices."""
    devices = []
    
    # Try ALSA devices first
    try:
        alsa_devices = await _get_alsa_devices()
        devices.extend(alsa_devices)
    except Exception:
        # Log but don't fail
        pass
    
    # Try JACK sources
    try:
        jack_devices = await _get_jack_sources()
        devices.extend(jack_devices)
    except Exception:
        # Log but don't fail
        pass
    
    # Add fallback options
    if not devices:
        devices = [
            ConfigValueOption("Default Audio Input", "default"),
            ConfigValueOption("Manual Entry (alsa:hw:X,Y)", "alsa:"),
            ConfigValueOption("Manual Entry (jack:port_name)", "jack:"),
        ]
    
    return devices


async def _get_alsa_devices() -> list[ConfigValueOption]:
    """Get ALSA capture devices."""
    devices = []
    
    try:
        # Use arecord to list capture devices
        returncode, output = await check_output("arecord", "-l")
        if returncode == 0:
            lines = output.decode('utf-8').strip().split('\n')
            for line in lines:
                if 'card' in line and 'device' in line:
                    # Parse line like: "card 1: USB [USB Audio], device 0: USB Audio [USB Audio]"
                    if 'card' in line and 'device' in line:
                        try:
                            # Extract card and device numbers
                            card_part = line.split('card ')[1].split(':')[0]
                            device_part = line.split('device ')[1].split(':')[0]
                            
                            # Extract friendly name
                            name_part = line.split(': ')[1] if ': ' in line else f"Card {card_part} Device {device_part}"
                            
                            devices.append(ConfigValueOption(
                                name_part,
                                f"alsa:hw:{card_part},{device_part}"
                            ))
                        except (IndexError, ValueError):
                            # Skip malformed lines
                            continue
    except Exception:
        # arecord not available or failed
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
        self.friendly_name: str = cast(str, self.config.get_value(CONF_FRIENDLY_NAME))
        self.thumbnail_image: str = cast(str, self.config.get_value(CONF_THUMBNAIL_IMAGE) or "")
        
        # Parse device string to determine format and device
        self.ffmpeg_format, self.ffmpeg_device = self._parse_device_string(self.device)

        # Runtime helpers
        self.cache_dir = os.path.join(self.mass.cache_path, self.instance_id)
        self.named_pipe = f"/tmp/{self.instance_id}"   # noqa: S108
        self._capture_proc: AsyncProcess | None = None
        self._runner_task: asyncio.Task | None = None          # type: ignore[type-arg]
        self._stop_called = False
        self._capture_started = asyncio.Event()
        self._on_unload_callbacks: list[Callable[..., None]] = [
            # Register the image resolution endpoint
            self.mass.streams.register_dynamic_route(
                f"/api/providers/{self.instance_id}/resolve_image",
                self._handle_resolve_image,
            ),
        ]

        # Static plugin-wide audio source definition
        metadata = PlayerMedia("Live Audio Input")
        
        # Add thumbnail image if configured
        if self.thumbnail_image:
            # Determine if it's a URL or relative path
            is_url = self.thumbnail_image.startswith(('http://', 'https://'))
            
            if is_url:
                # Direct URL - use image_url for PlayerMedia
                metadata.image_url = self.thumbnail_image
            else:
                # Relative path - resolve relative to provider directory
                provider_dir = os.path.dirname(__file__)
                image_path = os.path.join(provider_dir, self.thumbnail_image)
                
                # Check if file exists
                if os.path.exists(image_path):
                    # For local files, we need to create a URL that can be resolved
                    # Use the provider's resolve_image method via the webserver
                    metadata.image_url = f"/api/providers/{self.instance_id}/resolve_image?path={self.thumbnail_image}"
                else:
                    self.logger.warning("Thumbnail image not found: %s", image_path)
        
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
            metadata=metadata,
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
        self.logger.info("Unloading audio input provider %s", self.friendly_name)
        self._stop_called = True
        
        # Stop the capture process first
        if self._capture_proc and not self._capture_proc.closed:
            self.logger.info("Terminating capture process for %s", self.friendly_name)
            try:
                await self._capture_proc.close(True)  # Force kill
            except Exception as err:
                self.logger.warning("Error stopping capture process: %s", err)
        
        # Cancel the runner task
        if self._runner_task and not self._runner_task.done():
            self._runner_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._runner_task
        
        # Unregister callbacks
        for cb in self._on_unload_callbacks:
            try:
                cb()
            except Exception as err:
                self.logger.warning("Error during callback cleanup: %s", err)
        
        # Clean up the named pipe
        await self._cleanup_pipe()
        
        self.logger.info("Audio input provider %s unloaded successfully", self.friendly_name)

    # ---------------- PluginProvider hooks ----------------

    def get_source(self) -> PluginSource:
        """Expose the capture device as a PlayerSource."""
        return self._source_details

    async def resolve_image(self, path: str) -> str | bytes:
        """
        Resolve an image from an image path.

        This either returns (a generator to get) raw bytes of the image or
        a string with an http(s) URL or local path that is accessible from the server.
        """
        # For relative paths, resolve them relative to the provider directory
        if not path.startswith(('http://', 'https://')):
            provider_dir = os.path.dirname(__file__)
            full_path = os.path.join(provider_dir, path)
            if os.path.exists(full_path):
                return full_path
        
        # For URLs or if file doesn't exist, return as-is
        return path

    async def _handle_resolve_image(self, request) -> bytes:
        """Handle image resolution requests from the webserver."""
        from aiohttp.web import Response
        
        # Get the path parameter from the query string
        path = request.query.get('path', '')
        if not path:
            return Response(status=404, text="No path specified")
        
        try:
            # Use the resolve_image method to get the file path
            resolved_path = await self.resolve_image(path)
            
            if isinstance(resolved_path, str) and os.path.exists(resolved_path):
                # Read and return the file
                with open(resolved_path, 'rb') as f:
                    content = f.read()
                
                # Determine content type based on file extension
                content_type = "image/svg+xml" if resolved_path.endswith('.svg') else "image/png"
                
                return Response(body=content, content_type=content_type)
            else:
                return Response(status=404, text="Image not found")
                
        except Exception as err:
            self.logger.error("Error serving image %s: %s", path, err)
            return Response(status=500, text="Internal server error")

    # ---------------- Internals ----------------

    def _parse_device_string(self, device: str) -> tuple[str, str]:
        """Parse device string to determine FFmpeg format and device."""
        if device.startswith("pulse:"):
            # PulseAudio - but FFmpeg might not support it, try ALSA instead
            return "alsa", "default"
        elif device.startswith("jack:"):
            return "jack", device[5:]  # Remove "jack:" prefix
        elif device.startswith("alsa:"):
            return "alsa", device[5:]  # Remove "alsa:" prefix
        elif device == "default":
            return "alsa", "default"
        else:
            # Assume ALSA format
            return "alsa", device

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
        """Background task: keep audio capture alive using arecord + FFmpeg pipeline."""
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

        # Use arecord to capture audio and pipe to FFmpeg for processing
        # This avoids FFmpeg's input format issues
        capture_cmd: list[str] = [
            "sh", "-c",
            f"arecord -D {self.ffmpeg_device} -f S16_LE -c {self.channels} -r {self.sample_rate} -t raw | "
            f"ffmpeg -y -nostdin -hide_banner -loglevel warning "
            f"-f s16le -ac {self.channels} -ar {self.sample_rate} -i - "
            f"-acodec pcm_s16le -f s16le "
            f"-fflags +nobuffer -flags +low_delay "
            f"-probesize 32 -analyzeduration 0 "
            f"{self.named_pipe}"
        ]

        self.logger.info(
            "Launching audio capture pipeline for device: %s",
            self.ffmpeg_device,
        )

        retry_count = 0
        max_retries = 3

        while not self._stop_called and retry_count < max_retries:
            self._capture_proc = proc = AsyncProcess(
                capture_cmd, stdout=False, stderr=True, name=f"audio-capture[{self.friendly_name}]"
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
                    # Only log actual errors, not normal operational messages
                    if "broken pipe" in line.lower():
                        # Broken pipe is normal when no player is reading the stream
                        self.logger.debug("Broken pipe (normal when no player is reading): %s", line)
                    elif "overrun" in line.lower():
                        # Overruns are normal when no one is reading the stream
                        self.logger.debug("Audio buffer overrun (normal when stream not active): %s", line)
                    elif any(error_keyword in line.lower() for error_keyword in ['error', 'failed', 'cannot', 'unable']):
                        # Only log actual errors that aren't broken pipe related
                        if "broken pipe" not in line.lower():
                            self.logger.warning("Capture stderr: %s", line)
                    else:
                        # Log other messages at debug level
                        self.logger.debug("Capture info: %s", line)

                # Wait for process to complete and get return code
                return_code = await proc.wait()
                
                if return_code != 0:
                    self.logger.error(
                        "Capture process failed with return code %s. stderr output: %s",
                        return_code,
                        "\n".join(stderr_lines[-10:])  # Last 10 lines
                    )
                    
                    # Check if it's a device issue
                    stderr_text = "\n".join(stderr_lines)
                    if "No such file or directory" in stderr_text:
                        self.logger.error("Audio device '%s' not found", self.device)
                        break  # Don't retry for device issues
                    elif "Device or resource busy" in stderr_text:
                        self.logger.error("Device already in use: Audio device '%s' is currently being used by another application or Music Assistant instance. Please select a different audio input device.", self.device)
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
                # Clean up pipe before retry
                await self._cleanup_pipe()
                await asyncio.sleep(5)
                # Recreate pipe for next attempt
                try:
                    await check_output("mkfifo", self.named_pipe)
                except Exception as err:
                    self.logger.error("Failed to recreate named pipe %s: %s", self.named_pipe, err)
                    break
            else:
                self.logger.error("Max retries reached, stopping capture daemon")
                break

        # Final clean-up
        await self._cleanup_pipe()
