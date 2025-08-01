"""
Bluetooth Input Provider for Music Assistant.

This provider allows capturing audio from a locally connected Bluetooth receiver
and streaming it through the Music Assistant system as a Plugin Source for real-time audio.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from music_assistant_models.config_entries import ConfigEntry, ConfigValueType, ProviderConfig
from music_assistant_models.enums import (
    ConfigEntryType,
    ContentType,
    MediaType,
    ProviderFeature,
    StreamType,
)
from music_assistant_models.errors import MediaItemNotFoundError, ProviderUnavailableError
from music_assistant_models.media_items import AudioFormat, PluginSourceMediaItem
from music_assistant.helpers.process import AsyncProcess
from music_assistant.models.music_provider import MusicProvider
from music_assistant.models.plugin import PluginProvider, PluginSource

if TYPE_CHECKING:
    from music_assistant_models.media_items import BrowseFolder, MediaItem, SearchResults
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType


BLUETOOTH_INPUT_ID = "bluetooth_input"
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_CHANNELS = 2
DEFAULT_BIT_DEPTH = 16

# Configuration keys
CONF_AUDIO_DEVICE = "audio_device"
CONF_SAMPLE_RATE = "sample_rate"
CONF_CHANNELS = "channels"
CONF_BUFFER_SIZE = "buffer_size"
CONF_AUTO_START = "auto_start"


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    return BluetoothInputProvider(mass, manifest, config)


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
    return (
        ConfigEntry(
            key=CONF_AUDIO_DEVICE,
            type=ConfigEntryType.STRING,
            label="Audio Input Device",
            description="The audio input device to capture from (e.g., Bluetooth receiver)",
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
            description="Audio buffer size in milliseconds (lower = less delay, but may cause dropouts)",
            default_value=20,
            required=False,
        ),
        ConfigEntry(
            key=CONF_AUTO_START,
            type=ConfigEntryType.BOOLEAN,
            label="Auto Start",
            description="Automatically start capturing when provider loads",
            default_value=True,
            required=False,
        ),
    )


class BluetoothInputProvider(PluginProvider, MusicProvider):
    """Provider for capturing audio from Bluetooth input devices as a Plugin Source."""

    def __init__(self, mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig):
        """Initialize the provider."""
        super().__init__(mass, manifest, config)
        self._capture_process: AsyncProcess | None = None
        self._is_capturing = False
        self._capture_task: asyncio.Task | None = None
        self._plugin_source: PluginSource | None = None

    @property
    def supported_features(self) -> set[ProviderFeature]:
        """Return the features supported by this Provider."""
        return {
            ProviderFeature.AUDIO_SOURCE,
            ProviderFeature.BROWSE,
        }

    async def loaded_in_mass(self) -> None:
        """Call after the provider has been loaded."""
        await super().loaded_in_mass()
        
        # Create the plugin source
        sample_rate = self.config.get_value(CONF_SAMPLE_RATE)
        channels = self.config.get_value(CONF_CHANNELS)
        
        # Create a named pipe for real-time audio streaming
        import tempfile
        self._named_pipe = os.path.join(tempfile.gettempdir(), f"bluetooth_input_{self.instance_id}")
        
        self._plugin_source = PluginSource(
            id=self.instance_id,
            name="Bluetooth Audio Input",
            passive=False,
            # Kept `can_play_pause=False` and `can_seek=False` to maintain live streaming behavior
            can_play_pause=False,
            can_seek=False,
            audio_format=AudioFormat(
                content_type=ContentType.PCM_S16LE,
                sample_rate=sample_rate,
                bit_depth=DEFAULT_BIT_DEPTH,
                channels=channels,
            ),
            stream_type=StreamType.NAMED_PIPE,
            path=self._named_pipe,
        )
        
        self.logger.info("Created Bluetooth Audio Input source: %s", self._plugin_source.name)
        
        # Auto-start capturing if configured
        if self.config.get_value(CONF_AUTO_START):
            await self._start_capture()

    async def unload(self, is_removed: bool = False) -> None:
        """Handle unload/close of the provider."""
        await self._stop_capture()
        await super().unload(is_removed)

    def get_source(self) -> PluginSource:
        """Return the plugin source."""
        if not self._plugin_source:
            raise ProviderUnavailableError("Plugin source not initialized")
        return self._plugin_source


    async def _start_capture(self) -> None:
        """Start audio capture from the configured input device."""
        if self._is_capturing:
            return
        
        device = self.config.get_value(CONF_AUDIO_DEVICE)
        sample_rate = self.config.get_value(CONF_SAMPLE_RATE)
        channels = self.config.get_value(CONF_CHANNELS)
        buffer_size = self.config.get_value(CONF_BUFFER_SIZE)
        
        # Calculate period size for low latency (buffer_size in ms to samples)
        period_size = max(64, int(sample_rate * buffer_size / 1000))
        
        try:
            # Create named pipe if it doesn't exist
            if os.path.exists(self._named_pipe):
                os.remove(self._named_pipe)
            os.mkfifo(self._named_pipe)
            
            # Use direct arecord for minimal latency - output directly to named pipe
            # This eliminates the FFmpeg processing delay entirely
            command = [
                "arecord",
                "-D", device,
                "-f", "S16_LE",
                "-r", str(sample_rate),
                "-c", str(channels),
                "-t", "raw",
                "--buffer-size", str(period_size * 2),  # Minimal buffer for lowest latency
                "--period-size", str(period_size),
                self._named_pipe,  # Output directly to named pipe
            ]
            
            self.logger.info("Starting real-time audio capture from device: %s to pipe: %s", device, self._named_pipe)
            self._capture_process = AsyncProcess(
                command,
                stdin=False,
                stdout=False,  # No stdout since we're writing to named pipe
                stderr=True,
            )
            await self._capture_process.start()
            self._is_capturing = True
            self.logger.info("Started real-time audio capture from device: %s", device)
            
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
        
        # Clean up named pipe
        if hasattr(self, '_named_pipe') and os.path.exists(self._named_pipe):
            try:
                os.remove(self._named_pipe)
                self.logger.debug("Removed named pipe: %s", self._named_pipe)
            except Exception as err:
                self.logger.warning("Failed to remove named pipe: %s", err)
        
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
                        self.logger.error("Audio capture process died too many times (%d), stopping capture", restart_count)
                        # Log stderr for debugging
                        if self._capture_process and hasattr(self._capture_process, 'stderr'):
                            try:
                                stderr_data = await self._capture_process.stderr.read()
                                if stderr_data:
                                    self.logger.error("Audio capture stderr: %s", stderr_data.decode())
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

    # Music Provider methods for browsing functionality
    async def search(
        self,
        search_query: str,
        media_types: list[MediaType] | None = None,
        limit: int = 10,
    ) -> SearchResults:
        """Perform search on the provider."""
        # Return empty search results since this is just for browsing
        from music_assistant_models.media_items import SearchResults
        return SearchResults()

    async def browse_folder(
        self, path: str | None = None
    ) -> list[MediaItem | BrowseFolder]:
        """Browse folder."""
        if path is None or path == "":
            # Root level - return the Bluetooth input source
            bluetooth_source = PluginSourceMediaItem(
                item_id=self.instance_id,
                provider=self.instance_id,
                name="Bluetooth Audio Input",
                provider_mappings={
                    self.instance_id: self.instance_id
                },
                custom_data={"source_id": self.instance_id}
            )
            return [bluetooth_source]
        
        # No subfolders
        return []

    async def get_library_artists(self) -> AsyncGenerator[MediaItem, None]:
        """Retrieve library artists from the provider."""
        return
        yield  # pragma: no cover

    async def get_library_albums(self) -> AsyncGenerator[MediaItem, None]:
        """Retrieve library albums from the provider."""
        return
        yield  # pragma: no cover

    async def get_library_tracks(self) -> AsyncGenerator[MediaItem, None]:
        """Retrieve library tracks from the provider."""
        return
        yield  # pragma: no cover

    async def get_library_playlists(self) -> AsyncGenerator[MediaItem, None]:
        """Retrieve library playlists from the provider."""
        return
        yield  # pragma: no cover

    async def get_library_radios(self) -> AsyncGenerator[MediaItem, None]:
        """Retrieve library radio stations from the provider."""
        return
        yield  # pragma: no cover

    async def get_item(self, media_type: MediaType, item_id: str) -> MediaItem:
        """Get single media item by id."""
        if media_type == MediaType.PLUGIN_SOURCE and item_id == self.instance_id:
            return PluginSourceMediaItem(
                item_id=self.instance_id,
                provider=self.instance_id,
                name="Bluetooth Audio Input",
                provider_mappings={
                    self.instance_id: self.instance_id
                },
                custom_data={"source_id": self.instance_id}
            )
        
        raise MediaItemNotFoundError(f"Item {item_id} not found")
