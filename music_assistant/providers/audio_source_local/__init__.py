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
CONF_CUSTOM_IMAGE = "custom_image"
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
    # Get available audio devices and images for display
    device_info = await _get_audio_device_info()
    image_info = await _get_image_info()
    
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
            key="image_info_label",
            type=ConfigEntryType.LABEL,
            label=image_info,
        ),
        ConfigEntry(
            key=CONF_CUSTOM_IMAGE,
            type=ConfigEntryType.STRING,
            label="Custom Image",
            description="Enter one of the image names shown above, or a URL to an SVG image (leave blank for default icon.svg)",
            default_value="",
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
            
            device_info = "Available Audio Input Devices:\n• default – System default audio input"
            
            # Track devices we've already added to avoid duplicates
            added_devices = set()
            
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
                        
                        # Determine the best device identifier to show
                        if device_num == "0":
                            # For device 0, use simplified hw:card format
                            device_id = f"hw:{card_num}"
                            if device_desc and device_desc != card_description:
                                display_text = f"• {device_id} – {card_description} ({device_desc})"
                            else:
                                display_text = f"• {device_id} – {card_description}"
                        else:
                            # For other devices, use full hw:card,device format
                            device_id = f"hw:{card_num},{device_num}"
                            if device_desc and device_desc != card_description:
                                display_text = f"• {device_id} – {card_description} ({device_desc})"
                            else:
                                display_text = f"• {device_id} – {card_description}"
                        
                        # Only add if we haven't seen this device ID before
                        if device_id not in added_devices:
                            device_info += f"\n{display_text}"
                            added_devices.add(device_id)
                        
                    except (IndexError, ValueError):
                        continue
            
            if len(added_devices) == 0:  # No hardware devices found
                device_info += "\n⚠️  No hardware audio devices detected\n   Try: default or check your audio setup"
            
            return device_info
            
    except FileNotFoundError:
        return ("Available Audio Input Devices:\n"
                "⚠️  'arecord' command not found\n"
                "   Install alsa-utils package for device detection\n\n"
                "Common devices to try:\n"
                "• default – System default audio input\n"
                "• hw:0, hw:1, hw:2 – Hardware devices")
    except Exception as e:
        return ("Available Audio Input Devices:\n"
                f"⚠️  Error detecting devices: {str(e)}\n\n"
                "Common devices to try:\n"
                "• default – System default audio input\n"
                "• hw:0, hw:1, hw:2 – Hardware devices")


async def _get_image_info() -> str:
    """Get formatted information about available custom images."""
    import os
    
    # Get the path to the images folder
    provider_dir = os.path.dirname(__file__)
    images_dir = os.path.join(provider_dir, "images")
    
    image_list = ["Available Custom Images:"]
    
    try:
        if os.path.exists(images_dir):
            # Get all image files in the images folder
            image_files = []
            for filename in os.listdir(images_dir):
                if filename.lower().endswith(('.svg', '.png', '.jpg', '.jpeg', '.gif', '.webp')):
                    # Remove extension for display
                    name_without_ext = os.path.splitext(filename)[0]
                    image_files.append((name_without_ext, filename))
            
            # Sort alphabetically
            image_files.sort(key=lambda x: x[0].lower())
            
            # Add each image to the list
            for name_without_ext, filename in image_files:
                # Create a descriptive name based on the filename
                display_name = name_without_ext.replace('_', ' ').replace('-', ' ').title()
                image_list.append(f"• {name_without_ext.lower()} – {display_name} icon")
        
        if len(image_list) == 1:  # Only header, no images found
            image_list.append("⚠️  No custom images found in images folder")
    
    except Exception as e:
        image_list.append(f"⚠️  Error reading images folder: {str(e)}")
    
    # Always add the default option and URL info
    image_list.append("• default – Default provider icon (leave blank for default)")
    image_list.append("• URL – Enter any https:// URL to an SVG image")
    
    return "\n".join(image_list)


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
        
        # Get the custom image from config, fallback to default icon
        custom_image = self.config.get_value(CONF_CUSTOM_IMAGE)
        if not custom_image or custom_image.strip() == "":
            image_path = "icon.svg"
            is_remote_url = False
        else:
            # Dynamically find the image file in the images folder or handle URL
            image_path = await self._find_image_file(custom_image)
            is_remote_url = image_path.startswith(("http://", "https://"))
        
        # Create unique radio item ID for this instance to prevent name conflicts
        unique_radio_id = f"{AUDIO_SOURCE_ID}_{self.instance_id}"
        
        # Return the local audio input as a radio station
        yield Radio(
            item_id=unique_radio_id,
            provider=self.instance_id,
            name=custom_name,
            provider_mappings={
                ProviderMapping(
                    item_id=unique_radio_id,
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
                        path=image_path,
                        provider=self.domain,
                        remotely_accessible=is_remote_url,
                    )
                ]),
            ),
        )

    async def get_radio(self, prov_radio_id: str) -> Radio:
        """Get full radio details by id."""
        # Check if this is our unique radio ID for this instance
        expected_id = f"{AUDIO_SOURCE_ID}_{self.instance_id}"
        if prov_radio_id != expected_id:
            raise MediaNotFoundError(f"Radio {prov_radio_id} not found")
        
        # Return the radio from the library
        async for radio in self.get_library_radios():
            if radio.item_id == prov_radio_id:
                return radio
        
        raise MediaNotFoundError(f"Radio {prov_radio_id} not found")

    async def get_stream_details(self, item_id: str, media_type: MediaType) -> StreamDetails:
        """Get streamdetails for a track/radio."""
        # Check if this is our unique radio ID for this instance
        expected_id = f"{AUDIO_SOURCE_ID}_{self.instance_id}"
        if item_id != expected_id:
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
        # Check if this is our unique radio ID for this instance
        expected_id = f"{AUDIO_SOURCE_ID}_{self.instance_id}"
        if streamdetails.item_id != expected_id:
            raise MediaNotFoundError(f"Item {streamdetails.item_id} not found")
        
        # Create a new capture process for each stream to avoid concurrency issues
        device = self.config.get_value(CONF_AUDIO_DEVICE)
        sample_rate = self.config.get_value(CONF_SAMPLE_RATE)
        channels = self.config.get_value(CONF_CHANNELS)
        
        # Simple command that tries audio capture first, then falls back to silence
        command = (
            f"arecord -D {device} -f S16_LE -r {sample_rate} -c {channels} -t raw 2>/dev/null | "
            f"ffmpeg -f s16le -ar {sample_rate} -ac {channels} "
            f"-i - -acodec pcm_s16le -f s16le "
            f"-fflags +nobuffer -flags +low_delay -probesize 32 -analyzeduration 0 -"
        )
        
        # Fallback command that always works - generates silence if audio capture fails
        fallback_command = (
            f"ffmpeg -f lavfi -i anullsrc=channel_layout=stereo:sample_rate={sample_rate} "
            f"-f s16le -acodec pcm_s16le -ar {sample_rate} -ac {channels} "
            f"-fflags +nobuffer -flags +low_delay -"
        )
        
        stream_process = None
        try:
            # First try the actual audio capture
            self.logger.info("Starting audio stream with command: %s", command)
            stream_process = AsyncProcess(
                ["sh", "-c", command],
                stdin=False,
                stdout=True,
                stderr=True,
            )
            await stream_process.start()
            self.logger.info("Started audio stream from device: %s", device)
            
            # Track if we've received any data
            data_received = False
            chunk_count = 0
            
            # Stream audio data from this dedicated process
            async for chunk in stream_process.iter_any():
                if chunk:
                    data_received = True
                    chunk_count += 1
                    yield chunk
                    
                # If we haven't received data after a reasonable time, something's wrong
                if chunk_count > 100 and not data_received:
                    self.logger.warning("No audio data received, falling back to silence generation")
                    break
                    
        except Exception as err:
            self.logger.error("Failed to stream audio: %s", err)
            # Don't raise exception, fall back to silence generation
            
        finally:
            if stream_process and not stream_process.closed:
                await stream_process.close()
                self.logger.info("Closed audio stream process")
        
        # If we reach here and haven't yielded any data, generate silence
        if not data_received:
            self.logger.info("Falling back to silence generation")
            silence_process = None
            try:
                silence_process = AsyncProcess(
                    ["sh", "-c", fallback_command],
                    stdin=False,
                    stdout=True,
                    stderr=True,
                )
                await silence_process.start()
                self.logger.info("Started silence generation")
                
                async for chunk in silence_process.iter_any():
                    if chunk:
                        yield chunk
                        
            except Exception as err:
                self.logger.error("Failed to generate silence: %s", err)
                # As last resort, generate silence manually
                silence_chunk = b'\x00' * 4096  # 2048 samples of 16-bit stereo silence
                while True:
                    yield silence_chunk
                    await asyncio.sleep(0.1)  # ~100ms chunks
                    
            finally:
                if silence_process and not silence_process.closed:
                    await silence_process.close()

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

    async def _find_image_file(self, image_name: str) -> str:
        """Find the actual image file based on the user input (case-insensitive) or return URL if provided."""
        import os
        
        # Handle special cases
        if image_name.lower() in ("default", "icon"):
            return "icon.svg"
        
        # Check if it's a URL (starts with http:// or https://)
        if image_name.startswith(("http://", "https://")):
            return image_name  # Return URL as-is
        
        provider_dir = os.path.dirname(__file__)
        images_dir = os.path.join(provider_dir, "images")
        
        if not os.path.exists(images_dir):
            return "icon.svg"  # Fallback to default
        
        # Clean the input name (remove extension if provided, make lowercase)
        clean_name = image_name.replace(".svg", "").replace(".png", "").replace(".jpg", "").replace(".jpeg", "").replace(".gif", "").replace(".webp", "").lower()
        
        try:
            # Look for matching files in the images folder
            for filename in os.listdir(images_dir):
                if filename.lower().endswith(('.svg', '.png', '.jpg', '.jpeg', '.gif', '.webp')):
                    # Remove extension and compare
                    file_name_without_ext = os.path.splitext(filename)[0].lower()
                    if file_name_without_ext == clean_name:
                        return filename
        except Exception:
            pass
        
        # If no match found, fallback to default
        return "icon.svg"

    async def resolve_image(self, path: str) -> str | bytes:
        """Resolve an image from an image path or URL."""
        import os
        
        self.logger.debug("resolve_image called with path: %s", path)
        
        # Handle URLs - return as-is for Music Assistant to fetch
        if path.startswith(("http://", "https://")):
            self.logger.debug("Returning URL as-is: %s", path)
            return path
        
        provider_dir = os.path.dirname(__file__)
        
        # Handle custom images from the images folder (any image file)
        if path != "icon.svg":
            image_path = os.path.join(provider_dir, "images", path)
            if os.path.exists(image_path):
                self.logger.debug("Found local image: %s", image_path)
                return image_path
            else:
                self.logger.debug("Local image not found: %s", image_path)
        
        # Handle default provider icon
        if path == "icon.svg":
            icon_path = os.path.join(provider_dir, path)
            if os.path.exists(icon_path):
                self.logger.debug("Found default icon: %s", icon_path)
                return icon_path
            else:
                self.logger.debug("Default icon not found: %s", icon_path)
        
        self.logger.debug("Returning path unchanged: %s", path)
        return path
