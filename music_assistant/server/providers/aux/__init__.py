"""Local Sound Card AUX Provider

This creates a virtual radio output that streams sound output encoded via ffmpeg from 
the sound card to Music Assistant.
"""

from __future__ import annotations

import subprocess
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from music_assistant.common.models.config_entries import ConfigEntry
from music_assistant.common.models.media_items import Radio, ProviderMapping, StreamDetails, AudioFormat, ContentType, MediaType
from music_assistant.server.helpers.process import AsyncProcess
from music_assistant.common.models.enums import ConfigEntryType, ProviderFeature
from music_assistant.server.models.music_provider import MusicProvider

if TYPE_CHECKING:
    from music_assistant.common.models.config_entries import ProviderConfig
    from music_assistant.common.models.provider import ProviderManifest
    from music_assistant.server import MusicAssistant
    from music_assistant.server.models import ProviderInstanceType

FFMPEG_CMD = [
    "ffmpeg",
    "-f", "alsa",
    "-i", "default",
    "-f", "mp3",
    "-acodec", "libmp3lame",
    "-",
]

SUPPORTED_FEATURES = (
    ProviderFeature.LIBRARY_RADIOS,
)

async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    prov = AUXProvider(mass, manifest, config)
    await prov.handle_setup()
    return prov

async def get_config_entries(
    mass: MusicAssistant,
    instance_id: str | None = None,
    action: str | None = None,
    values: dict[str, ConfigValueType] | None = None,
) -> tuple[ConfigEntry, ...]:
    return tuple()

class AUXProvider(MusicProvider):

    @property
    def supported_features(self) -> tuple[ProviderFeature, ...]:
        """Return the features supported by this Provider."""
        return SUPPORTED_FEATURES

    async def handle_setup(self) -> None:
        pass

    async def get_library_radios(self) -> AsyncGenerator[Radio, None]:
        yield await self.get_radio("AUX")

    async def get_radio(self, prov_radio_id: str) -> Radio:
            return Radio(
                provider=self.domain,
                item_id="AUX",
                name="AUX",
                provider_mappings={
                    ProviderMapping(
                        item_id="AUX",
                        provider_domain=self.domain,
                        provider_instance=self.instance_id,
                    )
                },
            )

    async def get_stream_details(self, item_id: str) -> StreamDetails:
        return StreamDetails(
            provider=self.instance_id,
            item_id=item_id,
            content_type=ContentType.MP3,
            audio_format=AudioFormat(
                content_type=ContentType.MP3,
                sample_rate=44100,
                channels=2,
                bitrate=192,
            ),
            media_type=MediaType.RADIO,
            data="AUX",
        )

    async def get_audio_stream(self, streamdetails: StreamDetails) -> AsyncGenerator[bytes, None]:
        async with AsyncProcess(FFMPEG_CMD) as ffmpeg_proc:
            async for chunk in ffmpeg_proc.iter_any():
                yield chunk
