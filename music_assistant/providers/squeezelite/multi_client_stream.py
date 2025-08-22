"""Implementation of a simple multi-client stream task/job."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import suppress

from music_assistant_models.enums import ContentType
from music_assistant_models.media_items import AudioFormat

from music_assistant.helpers.audio import create_wave_header, get_ffmpeg_stream
from music_assistant.helpers.util import empty_queue

LOGGER = logging.getLogger(__name__)


class MultiClientStream:
    """Implementation of a simple multi-client (audio) stream task/job."""

    def __init__(
        self,
        audio_source: AsyncGenerator[bytes, None],
        audio_format: AudioFormat,
        expected_clients: int = 0,
        prefer_wav_fastpass: bool = False,
    ) -> None:
        """Initialize MultiClientStream."""
        self.audio_source = audio_source
        self.audio_format = audio_format
        self.subscribers: list[asyncio.Queue] = []
        self.expected_clients = expected_clients
        self.prefer_wav_fastpass = prefer_wav_fastpass
        self.task = asyncio.create_task(self._runner())

    @property
    def done(self) -> bool:
        """Return if this stream is already done."""
        return self.task.done()

    async def stop(self) -> None:
        """Stop/cancel the stream."""
        if self.done:
            return
        self.task.cancel()
        with suppress(asyncio.CancelledError):
            await self.task
        for sub_queue in list(self.subscribers):
            empty_queue(sub_queue)

    async def get_stream(
        self,
        output_format: AudioFormat,
        filter_params: list[str] | None = None,
    ) -> AsyncGenerator[bytes, None]:
        """Get (client specific encoded) ffmpeg stream."""
        # Enhanced WAV fast-path: bypass ffmpeg when output = WAV and formats match and no filters
        same_format = (
            (output_format.sample_rate or 0) == (self.audio_format.sample_rate or 0)
            and (output_format.bit_depth or 0) == (self.audio_format.bit_depth or 0)
            and (output_format.channels or 0) == (self.audio_format.channels or 0)
        )
        
        # Enable WAV fast-path if:
        # 1. Output format is WAV
        # 2. Audio formats match exactly
        # 3. No filters are applied
        # 4. Stream is marked as preferring WAV fast-path (for plugin sources)
        use_wav_fastpath = (
            output_format.content_type == ContentType.WAV
            and same_format
            and not filter_params
            and (self.prefer_wav_fastpass or self.audio_format.content_type.is_pcm())
        )
        
        if use_wav_fastpath:
            LOGGER.debug(
                "MultiClientStream: Using WAV fast-path (no ffmpeg) - format match: %s, prefer_wav: %s, no filters: %s",
                same_format, self.prefer_wav_fastpass, not filter_params
            )
            # write an unspecified-length WAV header then raw PCM chunks
            wav_header = create_wave_header(
                samplerate=output_format.sample_rate,
                channels=output_format.channels,
                bitspersample=output_format.bit_depth,
                duration=None,
            )
            yield wav_header

            # stream raw PCM directly (ultra-low latency path)
            async for chunk in self.subscribe_raw():
                yield chunk
            return

        # default ffmpeg path for format conversion / filtering
        LOGGER.debug(
            "MultiClientStream: Using ffmpeg path - format match: %s, prefer_wav: %s, filters: %s",
            same_format, self.prefer_wav_fastpass, filter_params
        )
        async for chunk in get_ffmpeg_stream(
            audio_input=self.subscribe_raw(),
            input_format=self.audio_format,
            output_format=output_format,
            filter_params=filter_params,
        ):
            yield chunk

    async def subscribe_raw(self) -> AsyncGenerator[bytes, None]:
        """Subscribe to the raw/unaltered audio stream."""
        try:
            queue = asyncio.Queue(2)
            self.subscribers.append(queue)
            while True:
                chunk = await queue.get()
                if chunk == b"":
                    break
                yield chunk
        finally:
            with suppress(ValueError):
                self.subscribers.remove(queue)

    async def _runner(self) -> None:
        """Run the stream for the given audio source."""
        # wait up to 5s for initial subscribers to attach, but be flexible about the count
        expected_clients = max(1, int(self.expected_clients or 1))
        count = 0
        max_wait_iterations = 50  # 5 seconds total wait time
        while count < max_wait_iterations:
            await asyncio.sleep(0.1)
            count += 1
            # Start streaming as soon as we have at least one client
            if len(self.subscribers) >= 1:
                # If we have the expected number, start immediately
                if len(self.subscribers) >= expected_clients:
                    break
                # If we have fewer than expected but waited at least 2 seconds, start anyway
                if count >= 20:  # 2 seconds
                    break
            # For groups of 3+, log progress every second
            if expected_clients >= 3 and count % 10 == 0:
                LOGGER.debug(
                    "Waiting for clients: %s/%s connected (waited %s ms)",
                    len(self.subscribers), expected_clients, count * 100
                )
        
        LOGGER.debug(
            "Starting multi-client stream with %s/%s clients (waited %s ms)",
            len(self.subscribers),
            expected_clients,
            count * 100,
        )
        
        # Track if we've started streaming to handle dynamic client changes
        stream_started = False
        
        async for chunk in self.audio_source:
            # Handle the case where all clients disconnect during streaming
            if len(self.subscribers) == 0:
                if not stream_started:
                    # Haven't started yet, wait a bit for clients
                    fail_count = 0
                    while len(self.subscribers) == 0 and fail_count < 50:  # 5 seconds
                        await asyncio.sleep(0.1)
                        fail_count += 1
                    
                    if len(self.subscribers) == 0:
                        LOGGER.warning(
                            "No clients connected after waiting, stopping stream (expected: %s)",
                            expected_clients
                        )
                        return
                else:
                    # Stream was running but all clients disconnected
                    # Wait a shorter time for reconnection (group membership change)
                    fail_count = 0
                    while len(self.subscribers) == 0 and fail_count < 30:  # 3 seconds
                        await asyncio.sleep(0.1)
                        fail_count += 1
                        
                        # Log every second during reconnection wait
                        if fail_count % 10 == 0:
                            LOGGER.debug(
                                "All clients disconnected, waiting for reconnection: %s seconds",
                                fail_count / 10
                            )
                    
                    if len(self.subscribers) == 0:
                        LOGGER.info(
                            "All clients disconnected and none reconnected, stopping stream"
                        )
                        return
                    else:
                        LOGGER.debug(
                            "Clients reconnected (%s), resuming stream", 
                            len(self.subscribers)
                        )
            
            # Mark that we've started streaming
            if not stream_started:
                stream_started = True
                LOGGER.debug("Multi-client stream started with %s clients", len(self.subscribers))
            
            # Send chunk to all connected subscribers
            await asyncio.gather(
                *[sub.put(chunk) for sub in self.subscribers], return_exceptions=True
            )
        
        # EOF: send empty chunk to remaining subscribers
        if self.subscribers:
            await asyncio.gather(*[sub.put(b"") for sub in self.subscribers], return_exceptions=True)
            LOGGER.debug("Multi-client stream ended, sent EOF to %s clients", len(self.subscribers))
