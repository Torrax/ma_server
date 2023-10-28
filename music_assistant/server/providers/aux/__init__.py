"""Local Sound Card AUX Provider

This creates a virtual radio output that streams sound output encoded via ffmpeg from 
the sound card to Music Assistant.
"""
from __future__ import annotations

import os
import subprocess
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from music_assistant.common.models.config_entries import ConfigEntry
from music_assistant.common.models.media_items import Radio, StreamDetails
from music_assistant.server.models.music_provider import MusicProvider

if TYPE_CHECKING:
    from music_assistant.common.models.config_entries import ProviderConfig
    from music_assistant.common.models.provider import ProviderManifest
    from music_assistant.server import MusicAssistant
    from music_assistant.server.models import ProviderInstanceType


FFMPEG_CMD = [
    "ffmpeg",
    "-f", "alsa",           # Audio format (ALSA)
    "-i", "default",        # Input source (default sound card)
    "-f", "mp3",            # Output format (MP3)
    "-acodec", "libmp3lame", # MP3 codec
    "-",                    # Pipe the output
]

async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    prov = AUXProvider(mass, manifest, config)
    await prov.handle_setup()
    return prov
    
async def get_config_entries(
    mass: MusicAssistant,
    instance_id: str | None = None,
    action: str | None = None,
    values: dict[str, ConfigValueType] | None = None,
) -> tuple[ConfigEntry, ...]:
    """
    Return Config entries to setup this provider.

    instance_id: id of an existing provider instance (None if new instance setup).
    action: [optional] action key called from config entries UI.
    values: the (intermediate) raw values for config entries sent with the action.
    """
    return tuple()  # we do not have any config entries (yet)

class AUXProvider(MusicProvider):
    radios = []

    async def handle_setup(self) -> None:
        """Enter Basic Setup Here"""
        # Create a virtual radio and add to the internal state
        radio_station = Radio(
            provider=self.instance_id,
            item_id="AUX",  # This can be a unique ID for the AUX radio station.
            title="AUX",
            description="AUX sound card input",
            image_url="",  # Placeholder, can be updated with an image URL if needed.
        )

    async def get_audio_stream(self, streamdetails: StreamDetails) -> AsyncGenerator[bytes, None]:
        """Return the audio stream for the AUX provider item."""
        proc = subprocess.Popen(FFMPEG_CMD, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        while True:
            chunk = proc.stdout.read(4096)  # Read in chunks of 4KB
            if not chunk:
                break
            yield chunk
