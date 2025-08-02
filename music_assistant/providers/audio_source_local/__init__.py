"""
Local Audio Source Provider for Music Assistant.

This provider allows capturing audio from locally connected audio sources
(Bluetooth, line-in, microphone, etc.) and streaming it through the Music Assistant system using FFmpeg.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from music_assistant_models.config_entries import ConfigEntry, ConfigValueType, ProviderConfig
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

if TYPE_CHECKING:
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType


AUDIO_SOURCE_ID = "audio_source_local"
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_CHANNELS = 2
DEFAULT_BIT_DEPTH = 16

# Configuration keys
CONF_CUSTOM_NAME = "custom_name"
CONF_AUDIO_DEVICE = "audio_device"
CONF_SAMPLE_RATE = "sample_rate"
CONF_CHANNELS = "channels"
CONF_BUFFER_SIZE = "buffer_size"
CONF_AUTO_START = "auto_start"


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    return LocalAudioSourceProvider(mass, manifest, config)


async def get_config_entries(
    mass: MusicAssistant,  # noqa: ARG001
    instance_id: str | None = None,  # noqa: ARG001
    action: str | None = None,  # noqa: ARG001
    values: dict[str, ConfigValueType] | None = None,  # noqa: ARG001
) -> tuple[ConfigEntry, ...]:
    """
    Return Config entries to setup this provider.

    instance_id: id of an existing provider instance (None if new instance setup).
    action: [optional] action key called from config entries UI.
    values: the (intermediate) raw values for config entries sent with the action.
    """
    # Get available audio devices for display
    device_info = await _get_audio_device_info()
    
    return (
        ConfigEntry(
            key=CONF_CUSTOM_NAME,
            type=ConfigEntryType.STRING,
            label="Custom Name",
            description="Custom name for this audio source (e.g., 'Living Room Bluetooth', 'Turntable')",
            default_value="Local Audio Source",
            required=False,
        ),
        ConfigEntry(
            key="device_info_label",
            type=ConfigEntryType.LABEL,
            label=device_info,
        ),
        ConfigEntry(
            key=CONF_AUDIO_DEVICE,
            type=ConfigEntryType.STRING,
            label="Audio Input Device",
            description="Enter one of the device identifiers shown above (e.g., hw:1,0, default, pulse)",
            default_value="default",
            required=True,
        ),
        ConfigEntry(
            key=CONF_SAMPLE_RATE,
            type=ConfigEntryType.INTEGER,
            label="Sample Rate",
            description="Audio sample rate in Hz",
            default_value=DEFAULT_SAMPLE_RATE,
            required=False,
        ),
        ConfigEntry(
            key=CONF_CHANNELS,
            type=ConfigEntryType.INTEGER,
            label="Channels",
            description="Number of audio channels",
            default_value=DEFAULT_CHANNELS,
            required=False,
        ),
        ConfigEntry(
            key=CONF_BUFFER_SIZE,
            type=ConfigEntryType.INTEGER,
            label="Buffer Size",
            description="Audio buffer size in milliseconds (lower = less delay)",
            default_value=50,
            required=False,
        ),
        ConfigEntry(
            key=CONF_AUTO_START,
            type=ConfigEntryType.BOOLEAN,
            label="Auto Start",
            description="Automatically start capturing when provider loads",
            default_value=False,
            required=False,
        ),
    )


async def _get_audio_device_info() -> str:
    """Get formatted information about available audio input devices."""
    try:
        # Run arecord -l to get capture devices
        proc = await asyncio.create_subprocess_exec(
            "arecord", "-l",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode == 0 and stdout:
            output = stdout.decode()
            lines = output.split('\n')
            
            device_info = ["📻 Available Audio Input Devices:"]
            device_info.append("")  # Empty line for spacing
            
            # Add common devices first
            device_info.append("• default - System default audio input")
            device_info.append("• pulse - PulseAudio device (if available)")
            device_info.append("")
            
            # Parse hardware devices
            for line in lines:
                if 'card' in line and 'device' in line:
                    try:
                        # Example: "card 1: CODEC [USB Audio CODEC], device 0: USB Audio [USB Audio]"
                        card_part = line.split('card ')[1]
                        card_num = card_part.split(':')[0].strip()
                        
                        # Extract card name (between : and [)
                        card_name_part = card_part.split(':')[1]
                        if '[' in card_name_part:
                            card_name = card_name_part.split('[')[0].strip()
                            card_description = card_name_part.split('[')[1].split(']')[0]
                        else:
                            card_name = card_name_part.strip()
                            card_description = card_name
                        
                        # Extract device number
                        device_part = line.split('device ')[1]
                        device_num = device_part.split(':')[0].strip()
                        
                        # Extract device description
                        device_desc_part = device_part.split(':')[1]
                        if '[' in device_desc_part:
                            device_desc = device_desc_part.split('[')[1].split(']')[0]
                        else:
                            device_desc = device_desc_part.strip()
                        
                        # Format device entries
                        device_id_full = f"hw:{card_num},{device_num}"
                        device_id_simple = f"hw:{card_num}"
                        
                        device_info.append(f"• {device_id_full} - {card_description} ({device_desc})")
                        if device_id_simple != device_id_full:
                            device_info.append(f"• {device_id_simple} - {card_description} (simplified)")
                        
                    except (IndexError, ValueError):
                        continue
            
            if len(device_info) <= 4:  # Only default entries, no hardware found
                device_info.append("⚠️  No hardware audio devices detected")
                device_info.append("   Try: default, pulse, or check your audio setup")
            
            return "\n".join(device_info)
            
    except FileNotFoundError:
        return ("📻 Audio Device Detection:\n\n"
                "⚠️  'arecord' command not found\n"
                "   Install alsa-utils package for device detection\n\n"
                "Common devices to try:\n"
                "• default - System default audio input\n"
                "• pulse - PulseAudio device\n"
                "• hw:0, hw:1, hw:2 - Hardware devices")
    except Exception as e:
        return ("📻 Audio Device Detection:\n\n"
                f"⚠️  Error detecting devices: {str(e)}\n\n"
                "Common devices to try:\n"
                "• default - System default audio input\n"
                "• pulse - PulseAudio device\n"
                "• hw:0, hw:1, hw:2 - Hardware devices")


async def _get_audio_devices() -> list[str]:
    """Get list of available audio input devices."""
    devices = ["default"]
    
    try:
        # Try to get audio devices using arecord (part of alsa-utils)
        proc = await asyncio.create_subprocess_exec(
            "arecord", "-l",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode == 0 and stdout:
            # Parse arecord output for capture devices
            output = stdout.decode()
            lines = output.split('\n')
            
            for line in lines:
                if 'card' in line and 'device' in line:
                    # Example: "card 1: USB [USB Audio], device 0: USB Audio [USB Audio]"
                    try:
                        # Extract card number and name
                        card_part = line.split('card ')[1]
                        card_num = card_part.split(':')[0].strip()
                        
                        # Extract device number
                        device_part = line.split('device ')[1]
                        device_num = device_part.split(':')[0].strip()
                        
                        device_id = f"hw:{card_num},{device_num}"
                        if device_id not in devices:
                            devices.append(device_id)
                        
                        # Also add simplified hw:card format
                        simple_device_id = f"hw:{card_num}"
                        if simple_device_id not in devices:
                            devices.append(simple_device_id)
                            
                    except (IndexError, ValueError):
                        continue
                                
    except Exception:
        # Fallback to common device names if detection fails
        pass
    
    # Add common fallback devices
    fallback_devices = [
        "hw:0",
        "hw:1", 
        "hw:2",
        "pulse",
        "plughw:0",
        "plughw:1",
    ]
    
    for device_id in fallback_devices:
        if device_id not in devices:
            devices.append(device_id)
    
    return devices


class LocalAudioSourceProvider(MusicProvider):
    """Provider for capturing audio from local audio sources (Bluetooth, line-in, microphone, etc.)."""

    def __init__(self, mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig):
        """Initialize the provider."""
        super().__init__(mass, manifest, config)
        self._capture_process: AsyncProcess | None = None
        self._is_capturing = False
        self._capture_task: asyncio.Task | None = None

    @property
    def supported_features(self) -> set[ProviderFeature]:
        """Return the features supported by this Provider."""
        return {
            ProviderFeature.BROWSE,
            ProviderFeature.LIBRARY_RADIOS,
        }

    @property
    def is_streaming_provider(self) -> bool:
        """Return True if the provider is a streaming provider."""
        return False

    async def loaded_in_mass(self) -> None:
        """Call after the provider has been loaded."""
        await super().loaded_in_mass()
        
        # Auto-start capturing if configured
        if self.config.get_value(CONF_AUTO_START):
            await self._start_capture()

    async def unload(self, is_removed: bool = False) -> None:
        """Handle unload/close of the provider."""
        await self._stop_capture()
        await super().unload(is_removed)

    async def get_library_radios(self) -> AsyncGenerator[Radio, None]:
        """Retrieve library/subscribed radio stations from the provider."""
        # Get the custom name from config, fallback to manifest name
        custom_name = self.config.get_value(CONF_CUSTOM_NAME)
        if not custom_name or custom_name.strip() == "":
            custom_name = "Local Audio Source"
        
        # Return the local audio input as a radio station
        yield Radio(
            item_id=AUDIO_SOURCE_ID,
            provider=self.instance_id,
            name=custom_name,
            provider_mappings={
                ProviderMapping(
                    item_id=AUDIO_SOURCE_ID,
                    provider_domain=self.domain,
                    provider_instance=self.instance_id,
                    available=True,
                    audio_format=AudioFormat(
                        content_type=ContentType.PCM_S16LE,
                        sample_rate=self.config.get_value(CONF_SAMPLE_RATE),
                        bit_depth=DEFAULT_BIT_DEPTH,
                        channels=self.config.get_value(CONF_CHANNELS),
                    ),
                )
            },
            metadata=MediaItemMetadata(
                description="Live audio input from local audio source",
                images=UniqueList([
                    MediaItemImage(
                        type=ImageType.THUMB,
                        path="icon.svg",
                        provider=self.domain,
                        remotely_accessible=False,
                    )
                ]),
            ),
        )

    async def get_radio(self, prov_radio_id: str) -> Radio:
        """Get full radio details by id."""
        if prov_radio_id != AUDIO_SOURCE_ID:
            raise MediaNotFoundError(f"Radio {prov_radio_id} not found")
        
        # Return the radio from the library
        async for radio in self.get_library_radios():
            if radio.item_id == prov_radio_id:
                return radio
        
        raise MediaNotFoundError(f"Radio {prov_radio_id} not found")

    async def get_stream_details(self, item_id: str, media_type: MediaType) -> StreamDetails:
        """Get streamdetails for a track/radio."""
        if item_id != AUDIO_SOURCE_ID:
            raise MediaNotFoundError(f"Item {item_id} not found")
        
        sample_rate = self.config.get_value(CONF_SAMPLE_RATE)
        channels = self.config.get_value(CONF_CHANNELS)
        
        return StreamDetails(
            provider=self.instance_id,
            item_id=item_id,
            audio_format=AudioFormat(
                content_type=ContentType.PCM_S16LE,
                sample_rate=sample_rate,
                bit_depth=DEFAULT_BIT_DEPTH,
                channels=channels,
            ),
            media_type=MediaType.RADIO,
            stream_type=StreamType.CUSTOM,
            can_seek=False,
            allow_seek=False,
        )

    async def get_audio_stream(
        self, streamdetails: StreamDetails, seek_position: int = 0
    ) -> AsyncGenerator[bytes, None]:
        """Return the audio stream for the local audio source."""
        if streamdetails.item_id != AUDIO_SOURCE_ID:
            raise MediaNotFoundError(f"Item {streamdetails.item_id} not found")
        
        # Start capturing if not already started
        if not self._is_capturing:
            await self._start_capture()
        
        # Stream audio data from the capture process
        if self._capture_process and not self._capture_process.closed:
            async for chunk in self._capture_process.iter_any():
                yield chunk
        else:
            raise ProviderUnavailableError("Audio capture process not available")

    async def _start_capture(self) -> None:
        """Start audio capture from the configured input device."""
        if self._is_capturing:
            return
        
        device = self.config.get_value(CONF_AUDIO_DEVICE)
        sample_rate = self.config.get_value(CONF_SAMPLE_RATE)
        channels = self.config.get_value(CONF_CHANNELS)
        buffer_size = self.config.get_value(CONF_BUFFER_SIZE)
        
        # Use arecord piped to ffmpeg with low-latency real-time streaming optimizations
        # Based on builtin_player implementation for near real-time performance
        command = (
            f"arecord -D {device} -f S16_LE -r {sample_rate} -c {channels} -t raw | "
            f"ffmpeg -f s16le -ar {sample_rate} -ac {channels} "
            f"-readrate 1.0 -readrate_initial_burst 0.5 "
            f"-i - -acodec pcm_s16le -f s16le "
            f"-fflags +nobuffer -flags +low_delay -probesize 32 -analyzeduration 0 -"
        )
        
        try:
            self.logger.info("Starting audio capture with command: %s", command)
            self._capture_process = AsyncProcess(
                ["sh", "-c", command],
                stdin=False,
                stdout=True,
                stderr=True,
            )
            await self._capture_process.start()
            self._is_capturing = True
            self.logger.info("Started audio capture from device: %s", device)
            
            # Start monitoring task
            self._capture_task = asyncio.create_task(self._monitor_capture())
            
        except Exception as err:
            self.logger.error("Failed to start audio capture: %s", err)
            await self._stop_capture()
            raise ProviderUnavailableError(f"Failed to start audio capture: {err}")

    async def _stop_capture(self) -> None:
        """Stop audio capture."""
        self._is_capturing = False
        
        if self._capture_task and not self._capture_task.done():
            self._capture_task.cancel()
            try:
                await self._capture_task
            except asyncio.CancelledError:
                pass
            self._capture_task = None
        
        if self._capture_process and not self._capture_process.closed:
            await self._capture_process.close()
            self._capture_process = None
        
        self.logger.info("Stopped audio capture")

    async def _monitor_capture(self) -> None:
        """Monitor the capture process and restart if needed."""
        restart_count = 0
        max_restarts = 3
        
        while self._is_capturing and self._capture_process:
            try:
                # Check if process is still running
                if self._capture_process.closed or self._capture_process.returncode is not None:
                    if restart_count >= max_restarts:
                        self.logger.error("FFmpeg process died too many times (%d), stopping capture", restart_count)
                        # Log stderr for debugging
                        if self._capture_process and hasattr(self._capture_process, 'stderr'):
                            try:
                                stderr_data = await self._capture_process.stderr.read()
                                if stderr_data:
                                    self.logger.error("FFmpeg stderr: %s", stderr_data.decode())
                            except Exception:
                                pass
                        await self._stop_capture()
                        break
                    
                    restart_count += 1
                    self.logger.warning("Capture process died (attempt %d/%d), attempting restart...", restart_count, max_restarts)
                    await self._stop_capture()
                    await asyncio.sleep(2)  # Longer delay before restart
                    await self._start_capture()
                else:
                    # Process is running, reset restart count
                    restart_count = 0
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as err:
                self.logger.error("Error in capture monitor: %s", err)
                await asyncio.sleep(5)

    async def resolve_image(self, path: str) -> str | bytes:
        """Resolve an image from an image path."""
        if path in ("icon.svg", "icon_monochrome.svg"):
            # Return the path to the actual icon file in our provider directory
            import os
            provider_dir = os.path.dirname(__file__)
            icon_path = os.path.join(provider_dir, path)
            if os.path.exists(icon_path):
                return icon_path
        return path
