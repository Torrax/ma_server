"""Model/base for a Music Provider implementation."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from music_assistant_models.enums import MediaType, ProviderFeature
from music_assistant_models.errors import (
    MediaNotFoundError,
    MusicAssistantError,
    UnsupportedFeaturedException,
)
from music_assistant_models.media_items import (
    Album,
    Artist,
    Audiobook,
    BrowseFolder,
    ItemMapping,
    MediaItemType,
    Playlist,
    Podcast,
    PodcastEpisode,
    Radio,
    RecommendationFolder,
    SearchResults,
    Track,
)

from music_assistant.constants import CACHE_CATEGORY_LIBRARY_ITEMS

from .provider import Provider

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from music_assistant_models.streamdetails import StreamDetails

    from music_assistant.controllers.media.albums import AlbumsController
    from music_assistant.controllers.media.artists import ArtistsController
    from music_assistant.controllers.media.audiobooks import AudiobooksController
    from music_assistant.controllers.media.playlists import PlaylistController
    from music_assistant.controllers.media.podcasts import PodcastsController
    from music_assistant.controllers.media.radio import RadioController
    from music_assistant.controllers.media.tracks import TracksController

# ruff: noqa: ARG001, ARG002


class MusicProvider(Provider):
    """Base representation of a Music Provider (controller).

    Music Provider implementations should inherit from this base model.
    """

    @property
    def is_streaming_provider(self) -> bool:
        """
        Return True if the provider is a streaming provider.

        This literally means that the catalog is not the same as the library contents.
        For local based providers (files, plex), the catalog is the same as the library content.
        It also means that data is if this provider is NOT a streaming provider,
        data cross instances is unique, the catalog and library differs per instance.

        Setting this to True will only query one instance of the provider for search and lookups.
        Setting this to False will query all instances of this provider for search and lookups.
        """
        return True

    @property
    def lookup_key(self) -> str:
        """Return domain if (multi-instance) streaming_provider or instance_id otherwise."""
        if self.is_streaming_provider or not self.manifest.multi_instance:
            return self.domain
        return self.instance_id

    async def loaded_in_mass(self) -> None:
        """Call after the provider has been loaded."""

    async def search(
        self,
        search_query: str,
        media_types: list[MediaType],
        limit: int = 5,
    ) -> SearchResults:
        """Perform search on musicprovider.

        :param search_query: Search query.
        :param media_types: A list of media_types to include.
        :param limit: Number of items to return in the search (per type).
        """
        if ProviderFeature.SEARCH in self.supported_features:
            raise NotImplementedError
        return SearchResults()

    async def get_library_artists(self) -> AsyncGenerator[Artist, None]:
        """Retrieve library artists from the provider."""
        yield  # type: ignore[misc]
        raise NotImplementedError

    async def get_library_albums(self) -> AsyncGenerator[Album, None]:
        """Retrieve library albums from the provider."""
        yield  # type: ignore[misc]
        raise NotImplementedError

    async def get_library_tracks(self) -> AsyncGenerator[Track, None]:
        """Retrieve library tracks from the provider."""
        yield  # type: ignore[misc]
        raise NotImplementedError

    async def get_library_playlists(self) -> AsyncGenerator[Playlist, None]:
        """Retrieve library/subscribed playlists from the provider."""
        yield  # type: ignore[misc]
        raise NotImplementedError

    async def get_library_radios(self) -> AsyncGenerator[Radio, None]:
        """Retrieve library/subscribed radio stations from the provider."""
        yield  # type: ignore[misc]
        raise NotImplementedError

    async def get_library_audiobooks(self) -> AsyncGenerator[Audiobook, None]:
        """Retrieve library/subscribed audiobooks from the provider."""
        yield  # type: ignore[misc]
        raise NotImplementedError

    async def get_library_podcasts(self) -> AsyncGenerator[Podcast, None]:
        """Retrieve library/subscribed podcasts from the provider."""
        yield  # type: ignore[misc]
        raise NotImplementedError

    async def get_artist(self, prov_artist_id: str) -> Artist:
        """Get full artist details by id."""
        raise NotImplementedError

    async def get_artist_albums(self, prov_artist_id: str) -> list[Album]:
        """Get a list of all albums for the given artist.

        Only called if provider supports ProviderFeature.ARTIST_ALBUMS.
        """
        raise NotImplementedError

    async def get_artist_toptracks(self, prov_artist_id: str) -> list[Track]:
        """Get a list of most popular tracks for the given artist.

        Only called if provider supports ProviderFeature.ARTIST_TOPTRACKS.
        """
        raise NotImplementedError

    async def get_album(self, prov_album_id: str) -> Album:
        """Get full album details by id.

        Only called if provider supports ProviderFeature.LIBRARY_ALBUMS.
        """
        raise NotImplementedError

    async def get_track(self, prov_track_id: str) -> Track:
        """Get full track details by id.

        Only called if provider supports ProviderFeature.LIBRARY_TRACKS.
        """
        raise NotImplementedError

    async def get_playlist(self, prov_playlist_id: str) -> Playlist:
        """Get full playlist details by id.

        Only called if provider supports ProviderFeature.LIBRARY_PLAYLISTS.
        """
        raise NotImplementedError

    async def get_radio(self, prov_radio_id: str) -> Radio:
        """Get full radio details by id.

        Only called if provider supports ProviderFeature.LIBRARY_RADIOS.
        """
        raise NotImplementedError

    async def get_audiobook(self, prov_audiobook_id: str) -> Audiobook:
        """Get full audiobook details by id.

        Only called if provider supports ProviderFeature.LIBRARY_AUDIOBOOKS.
        """
        raise NotImplementedError

    async def get_podcast(self, prov_podcast_id: str) -> Podcast:
        """Get full podcast details by id.

        Only called if provider supports ProviderFeature.LIBRARY_PODCASTS.
        """
        raise NotImplementedError

    async def get_podcast_episode(self, prov_episode_id: str) -> PodcastEpisode:
        """Get (full) podcast episode details by id.

        Only called if provider supports ProviderFeature.LIBRARY_PODCASTS.
        """
        raise NotImplementedError

    async def get_album_tracks(
        self,
        prov_album_id: str,
    ) -> list[Track]:
        """Get album tracks for given album id.

        Only called if provider supports ProviderFeature.LIBRARY_ALBUMS.
        """
        raise NotImplementedError

    async def get_playlist_tracks(
        self,
        prov_playlist_id: str,
        page: int = 0,
    ) -> list[Track]:
        """Get all playlist tracks for given playlist id.

        Only called if provider supports ProviderFeature.LIBRARY_PLAYLISTS.
        """
        raise NotImplementedError

    async def get_podcast_episodes(
        self,
        prov_podcast_id: str,
    ) -> AsyncGenerator[PodcastEpisode, None]:
        """Get all PodcastEpisodes for given podcast id.

        Only called if provider supports ProviderFeature.LIBRARY_PODCASTS.
        """
        yield  # type: ignore[misc]
        raise NotImplementedError

    async def library_add(self, item: MediaItemType) -> bool:
        """Add item to provider's library. Return true on success."""
        if (
            item.media_type == MediaType.ARTIST
            and ProviderFeature.LIBRARY_ARTISTS_EDIT in self.supported_features
        ):
            raise NotImplementedError
        if (
            item.media_type == MediaType.ALBUM
            and ProviderFeature.LIBRARY_ALBUMS_EDIT in self.supported_features
        ):
            raise NotImplementedError
        if (
            item.media_type == MediaType.TRACK
            and ProviderFeature.LIBRARY_TRACKS_EDIT in self.supported_features
        ):
            raise NotImplementedError
        if (
            item.media_type == MediaType.PLAYLIST
            and ProviderFeature.LIBRARY_PLAYLISTS_EDIT in self.supported_features
        ):
            raise NotImplementedError
        if (
            item.media_type == MediaType.RADIO
            and ProviderFeature.LIBRARY_RADIOS_EDIT in self.supported_features
        ):
            raise NotImplementedError
        if (
            item.media_type == MediaType.AUDIOBOOK
            and ProviderFeature.LIBRARY_AUDIOBOOKS_EDIT in self.supported_features
        ):
            raise NotImplementedError
        if (
            item.media_type == MediaType.PODCAST
            and ProviderFeature.LIBRARY_PODCASTS_EDIT in self.supported_features
        ):
            raise NotImplementedError
        self.logger.info(
            "Provider %s does not support library edit, "
            "the action will only be performed in the local database.",
            self.name,
        )
        return True

    async def library_remove(self, prov_item_id: str, media_type: MediaType) -> bool:
        """Remove item from provider's library. Return true on success."""
        if (
            media_type == MediaType.ARTIST
            and ProviderFeature.LIBRARY_ARTISTS_EDIT in self.supported_features
        ):
            raise NotImplementedError
        if (
            media_type == MediaType.ALBUM
            and ProviderFeature.LIBRARY_ALBUMS_EDIT in self.supported_features
        ):
            raise NotImplementedError
        if (
            media_type == MediaType.TRACK
            and ProviderFeature.LIBRARY_TRACKS_EDIT in self.supported_features
        ):
            raise NotImplementedError
        if (
            media_type == MediaType.PLAYLIST
            and ProviderFeature.LIBRARY_PLAYLISTS_EDIT in self.supported_features
        ):
            raise NotImplementedError
        if (
            media_type == MediaType.RADIO
            and ProviderFeature.LIBRARY_RADIOS_EDIT in self.supported_features
        ):
            raise NotImplementedError
        if (
            media_type == MediaType.AUDIOBOOK
            and ProviderFeature.LIBRARY_AUDIOBOOKS_EDIT in self.supported_features
        ):
            raise NotImplementedError
        if (
            media_type == MediaType.PODCAST
            and ProviderFeature.LIBRARY_PODCASTS_EDIT in self.supported_features
        ):
            raise NotImplementedError
        self.logger.info(
            "Provider %s does not support library edit, "
            "the action will only be performed in the local database.",
            self.name,
        )
        return True

    async def add_playlist_tracks(self, prov_playlist_id: str, prov_track_ids: list[str]) -> None:
        """Add track(s) to playlist.

        Only called if provider supports ProviderFeature.PLAYLIST_TRACKS_EDIT.
        """
        raise NotImplementedError

    async def remove_playlist_tracks(
        self, prov_playlist_id: str, positions_to_remove: tuple[int, ...]
    ) -> None:
        """Remove track(s) from playlist.

        Only called if provider supports ProviderFeature.PLAYLIST_TRACKS_EDIT.
        """
        raise NotImplementedError

    async def create_playlist(self, name: str) -> Playlist:
        """Create a new playlist on provider with given name.

        Only called if provider supports ProviderFeature.PLAYLIST_CREATE.
        """
        raise NotImplementedError

    async def get_similar_tracks(self, prov_track_id: str, limit: int = 25) -> list[Track]:
        """Retrieve a dynamic list of similar tracks based on the provided track.

        Only called if provider supports ProviderFeature.SIMILAR_TRACKS.
        """
        raise NotImplementedError

    async def get_resume_position(self, item_id: str, media_type: MediaType) -> tuple[bool, int]:
        """
        Get progress (resume point) details for the given Audiobook or Podcast episode.

        This is a separate call from the regular get_item call to ensure the resume position
        is always up-to-date and because a lot providers have this info present on a dedicated
        endpoint.

        Will be called right before playback starts to ensure the resume position is correct.

        Returns a boolean with the fully_played status
        and an integer with the resume position in ms.
        """
        raise NotImplementedError

    async def get_stream_details(self, item_id: str, media_type: MediaType) -> StreamDetails:
        """Get streamdetails for a track/radio/chapter/episode."""
        raise NotImplementedError

    async def get_audio_stream(
        self, streamdetails: StreamDetails, seek_position: int = 0
    ) -> AsyncGenerator[bytes, None]:
        """
        Return the (custom) audio stream for the provider item.

        Will only be called when the stream_type is set to CUSTOM.
        """
        yield b""
        raise NotImplementedError

    async def on_streamed(
        self,
        streamdetails: StreamDetails,
    ) -> None:
        """
        Handle callback when given streamdetails completed streaming.

        To get the number of seconds streamed, see streamdetails.seconds_streamed.
        To get the number of seconds seeked/skipped, see streamdetails.seek_position.
        Note that seconds_streamed is the total streamed seconds, so without seeked time.

        NOTE: Due to internal and player buffering,
        this may be called in advance of the actual completion.
        """

    async def on_played(
        self,
        media_type: MediaType,
        prov_item_id: str,
        fully_played: bool,
        position: int,
        media_item: MediaItemType,
        is_playing: bool = False,
    ) -> None:
        """
        Handle callback when a (playable) media item has been played.

        This is called by the Queue controller when;
            - a track has been fully played
            - a track has been stopped (or skipped) after being played
            - every 30s when a track is playing

        Fully played is True when the track has been played to the end.

        Position is the last known position of the track in seconds, to sync resume state.
        When fully_played is set to false and position is 0,
        the user marked the item as unplayed in the UI.

        media_item is the full media item details of the played/playing track.

        is_playing is True when the track is currently playing.
        """

    async def resolve_image(self, path: str) -> str | bytes:
        """
        Resolve an image from an image path.

        This either returns (a generator to get) raw bytes of the image or
        a string with an http(s) URL or local path that is accessible from the server.
        """
        return path

    async def get_item(self, media_type: MediaType, prov_item_id: str) -> MediaItemType:
        """Get single MediaItem from provider."""
        if media_type == MediaType.ARTIST:
            return await self.get_artist(prov_item_id)
        if media_type == MediaType.ALBUM:
            return await self.get_album(prov_item_id)
        if media_type == MediaType.PLAYLIST:
            return await self.get_playlist(prov_item_id)
        if media_type == MediaType.RADIO:
            return await self.get_radio(prov_item_id)
        if media_type == MediaType.AUDIOBOOK:
            return await self.get_audiobook(prov_item_id)
        if media_type == MediaType.PODCAST:
            return await self.get_podcast(prov_item_id)
        if media_type == MediaType.PODCAST_EPISODE:
            return await self.get_podcast_episode(prov_item_id)
        return await self.get_track(prov_item_id)

    async def browse(self, path: str) -> Sequence[MediaItemType | ItemMapping | BrowseFolder]:  # noqa: PLR0911
        """Browse this provider's items.

        :param path: The path to browse, (e.g. provider_id://artists).
        """
        if ProviderFeature.BROWSE not in self.supported_features:
            # we may NOT use the default implementation if the provider does not support browse
            raise NotImplementedError

        subpath = path.split("://", 1)[1]
        # this reference implementation can be overridden with a provider specific approach
        if subpath == "artists":
            library_item_ids = await self.mass.cache.get(
                "artist",
                category=CACHE_CATEGORY_LIBRARY_ITEMS,
                base_key=self.instance_id,
            )
            if not library_item_ids:
                return [x async for x in self.get_library_artists()]
            library_items = cast("list[int]", library_item_ids)
            query = "artists.item_id in :ids"
            query_params = {"ids": library_items}
            return await self.mass.music.artists.library_items(
                provider=self.instance_id,
                extra_query=query,
                extra_query_params=query_params,
            )
        if subpath == "albums":
            library_item_ids = await self.mass.cache.get(
                "album",
                category=CACHE_CATEGORY_LIBRARY_ITEMS,
                base_key=self.instance_id,
            )
            if not library_item_ids:
                return [x async for x in self.get_library_albums()]
            library_item_ids = cast("list[int]", library_item_ids)
            query = "albums.item_id in :ids"
            query_params = {"ids": library_item_ids}
            return await self.mass.music.albums.library_items(
                extra_query=query, extra_query_params=query_params
            )
        if subpath == "tracks":
            library_item_ids = await self.mass.cache.get(
                "track",
                category=CACHE_CATEGORY_LIBRARY_ITEMS,
                base_key=self.instance_id,
            )
            if not library_item_ids:
                return [x async for x in self.get_library_tracks()]
            library_item_ids = cast("list[int]", library_item_ids)
            query = "tracks.item_id in :ids"
            query_params = {"ids": library_item_ids}
            return await self.mass.music.tracks.library_items(
                extra_query=query, extra_query_params=query_params
            )
        if subpath == "radios":
            library_item_ids = await self.mass.cache.get(
                "radio",
                category=CACHE_CATEGORY_LIBRARY_ITEMS,
                base_key=self.instance_id,
            )
            if not library_item_ids:
                return [x async for x in self.get_library_radios()]
            library_item_ids = cast("list[int]", library_item_ids)
            query = "radios.item_id in :ids"
            query_params = {"ids": library_item_ids}
            return await self.mass.music.radio.library_items(
                extra_query=query, extra_query_params=query_params
            )
        if subpath == "playlists":
            library_item_ids = await self.mass.cache.get(
                "playlist",
                category=CACHE_CATEGORY_LIBRARY_ITEMS,
                base_key=self.instance_id,
            )
            if not library_item_ids:
                return [x async for x in self.get_library_playlists()]
            library_item_ids = cast("list[int]", library_item_ids)
            query = "playlists.item_id in :ids"
            query_params = {"ids": library_item_ids}
            return await self.mass.music.playlists.library_items(
                extra_query=query, extra_query_params=query_params
            )
        if subpath == "audiobooks":
            library_item_ids = await self.mass.cache.get(
                "audiobook",
                category=CACHE_CATEGORY_LIBRARY_ITEMS,
                base_key=self.instance_id,
            )
            if not library_item_ids:
                return [x async for x in self.get_library_audiobooks()]
            library_item_ids = cast("list[int]", library_item_ids)
            query = "audiobooks.item_id in :ids"
            query_params = {"ids": library_item_ids}
            return await self.mass.music.audiobooks.library_items(
                extra_query=query, extra_query_params=query_params
            )
        if subpath == "podcasts":
            library_item_ids = await self.mass.cache.get(
                "podcast",
                category=CACHE_CATEGORY_LIBRARY_ITEMS,
                base_key=self.instance_id,
            )
            if not library_item_ids:
                return [x async for x in self.get_library_podcasts()]
            library_item_ids = cast("list[int]", library_item_ids)
            query = "podcasts.item_id in :ids"
            query_params = {"ids": library_item_ids}
            return await self.mass.music.podcasts.library_items(
                extra_query=query, extra_query_params=query_params
            )
        if subpath:
            # unknown path
            msg = "Invalid subpath"
            raise KeyError(msg)

        # no subpath: return main listing
        folders: list[BrowseFolder] = []
        if ProviderFeature.LIBRARY_ARTISTS in self.supported_features:
            folders.append(
                BrowseFolder(
                    item_id="artists",
                    provider=self.instance_id,
                    path=path + "artists",
                    name="Artists",
                    translation_key="artists",
                    is_playable=True,
                )
            )
        if ProviderFeature.LIBRARY_ALBUMS in self.supported_features:
            folders.append(
                BrowseFolder(
                    item_id="albums",
                    provider=self.instance_id,
                    path=path + "albums",
                    name="Albums",
                    translation_key="albums",
                    is_playable=True,
                )
            )
        if ProviderFeature.LIBRARY_TRACKS in self.supported_features:
            folders.append(
                BrowseFolder(
                    item_id="tracks",
                    provider=self.domain,
                    path=path + "tracks",
                    name="Tracks",
                    translation_key="tracks",
                    is_playable=True,
                )
            )
        if ProviderFeature.LIBRARY_PLAYLISTS in self.supported_features:
            folders.append(
                BrowseFolder(
                    item_id="playlists",
                    provider=self.instance_id,
                    path=path + "playlists",
                    name="Playlists",
                    translation_key="playlists",
                    is_playable=True,
                )
            )
        if ProviderFeature.LIBRARY_RADIOS in self.supported_features:
            folders.append(
                BrowseFolder(
                    item_id="radios",
                    provider=self.instance_id,
                    path=path + "radios",
                    name="Radio",
                    translation_key="radios",
                )
            )
        if ProviderFeature.LIBRARY_AUDIOBOOKS in self.supported_features:
            folders.append(
                BrowseFolder(
                    item_id="audiobooks",
                    provider=self.instance_id,
                    path=path + "audiobooks",
                    name="Audiobooks",
                    translation_key="audiobooks",
                )
            )
        if ProviderFeature.LIBRARY_PODCASTS in self.supported_features:
            folders.append(
                BrowseFolder(
                    item_id="podcasts",
                    provider=self.instance_id,
                    path=path + "podcasts",
                    name="Podcasts",
                    translation_key="podcasts",
                )
            )
        if len(folders) == 1:
            # only one level, return the items directly
            return await self.browse(folders[0].path)
        return folders

    async def recommendations(self) -> list[RecommendationFolder]:
        """
        Get this provider's recommendations.

        Returns an actual (and often personalised) list of recommendations
        from this provider for the user/account.
        """
        if ProviderFeature.RECOMMENDATIONS in self.supported_features:
            raise NotImplementedError
        return []

    async def sync_library(self, media_type: MediaType) -> None:
        """Run library sync for this provider."""
        # ruff: noqa: PLR0915 # too many statements
        # this reference implementation can be overridden
        # with a provider specific approach if needed

        async def _controller_update_item_in_library(
            controller: ArtistsController
            | AlbumsController
            | TracksController
            | RadioController
            | PlaylistController
            | AudiobooksController
            | PodcastsController,
            prov_item: MediaItemType,
            item_id: str | int,
        ) -> Artist | Album | Track | Radio | Playlist | Audiobook | Podcast:
            """Update media item in controller including type checking.

            all isinstance(...) for type checking. The statement
            library_item = await controller.update_item_in_library(prov_item)
            cannot be moved out of this scope.
            """
            library_item: Artist | Album | Track | Radio | Playlist | Audiobook | Podcast
            if TYPE_CHECKING:
                if isinstance(prov_item, Artist):
                    assert isinstance(controller, ArtistsController)
                    library_item = await controller.update_item_in_library(item_id, prov_item)
                elif isinstance(prov_item, Album):
                    assert isinstance(controller, AlbumsController)
                    library_item = await controller.update_item_in_library(item_id, prov_item)
                elif isinstance(prov_item, Track):
                    assert isinstance(controller, TracksController)
                    library_item = await controller.update_item_in_library(item_id, prov_item)
                elif isinstance(prov_item, Radio):
                    assert isinstance(controller, RadioController)
                    library_item = await controller.update_item_in_library(item_id, prov_item)
                elif isinstance(prov_item, Playlist):
                    assert isinstance(controller, PlaylistController)
                    library_item = await controller.update_item_in_library(item_id, prov_item)
                elif isinstance(prov_item, Audiobook):
                    assert isinstance(controller, AudiobooksController)
                    library_item = await controller.update_item_in_library(item_id, prov_item)
                elif isinstance(prov_item, Podcast):
                    assert isinstance(controller, PodcastsController)
                    library_item = await controller.update_item_in_library(item_id, prov_item)
                else:
                    raise TypeError("Prov item unknown in this context.")
                return library_item
            else:
                return await controller.update_item_in_library(item_id, prov_item)

        if not self.library_supported(media_type):
            raise UnsupportedFeaturedException("Library sync not supported for this media type")
        self.logger.debug("Start sync of %s items.", media_type.value)
        controller = self.mass.music.get_controller(media_type)
        cur_db_ids = set()
        async for prov_item in self._get_library_gen(media_type):
            library_item = await controller.get_library_item_by_prov_mappings(
                prov_item.provider_mappings,
            )
            assert not isinstance(prov_item, PodcastEpisode)
            try:
                if not library_item and not prov_item.available:
                    # skip unavailable tracks
                    self.logger.debug(
                        "Skipping sync of item %s because it is unavailable",
                        prov_item.uri,
                    )
                    continue
                if not library_item:
                    # create full db item
                    # note that we skip the metadata lookup purely to speed up the sync
                    # the additional metadata is then lazy retrieved afterwards
                    if self.is_streaming_provider:
                        prov_item.favorite = True

                    # all isinstance(...) for type checking. The statement
                    # library_item = await controller.add_item_to_library(prov_item)
                    # cannot be moved out of this scope.
                    if TYPE_CHECKING:
                        if isinstance(prov_item, Artist):
                            assert isinstance(controller, ArtistsController)
                            library_item = await controller.add_item_to_library(prov_item)
                        elif isinstance(prov_item, Album):
                            assert isinstance(controller, AlbumsController)
                            library_item = await controller.add_item_to_library(prov_item)
                        elif isinstance(prov_item, Track):
                            assert isinstance(controller, TracksController)
                            library_item = await controller.add_item_to_library(prov_item)
                        elif isinstance(prov_item, Radio):
                            assert isinstance(controller, RadioController)
                            library_item = await controller.add_item_to_library(prov_item)
                        elif isinstance(prov_item, Playlist):
                            assert isinstance(controller, PlaylistController)
                            library_item = await controller.add_item_to_library(prov_item)
                        elif isinstance(prov_item, Audiobook):
                            assert isinstance(controller, AudiobooksController)
                            library_item = await controller.add_item_to_library(prov_item)
                        elif isinstance(prov_item, Podcast):
                            assert isinstance(controller, PodcastsController)
                            library_item = await controller.add_item_to_library(prov_item)
                        else:
                            raise RuntimeError
                    else:
                        library_item = await controller.add_item_to_library(prov_item)
                elif getattr(library_item, "cache_checksum", None) != getattr(
                    prov_item, "cache_checksum", None
                ):
                    # existing dbitem checksum changed (playlists only)
                    if TYPE_CHECKING:
                        assert isinstance(prov_item, Playlist)
                        assert isinstance(controller, PlaylistController)
                    library_item = await controller.update_item_in_library(
                        library_item.item_id, prov_item
                    )
                if library_item.available != prov_item.available:
                    # existing item availability changed
                    library_item = await _controller_update_item_in_library(
                        controller, prov_item, library_item.item_id
                    )
                # check if resume_position_ms or fully_played changed (audiobook only)
                resume_pos_prov = getattr(prov_item, "resume_position_ms", None)
                fully_played_prov = getattr(prov_item, "fully_played", None)
                if (
                    resume_pos_prov is not None
                    and fully_played_prov is not None
                    and (
                        getattr(library_item, "resume_position_ms", None) != resume_pos_prov
                        or getattr(library_item, "fully_played", None) != fully_played_prov
                    )
                ):
                    library_item = await _controller_update_item_in_library(
                        controller, prov_item, library_item.item_id
                    )

                cur_db_ids.add(int(library_item.item_id))
                await asyncio.sleep(0)  # yield to eventloop
            except MusicAssistantError as err:
                self.logger.warning(
                    "Skipping sync of item %s - error details: %s",
                    prov_item.uri,
                    str(err),
                )

        # process deletions (= no longer in library)
        cache_category = CACHE_CATEGORY_LIBRARY_ITEMS
        cache_base_key = self.instance_id

        prev_library_items: list[int] | None
        if prev_library_items := await self.mass.cache.get(
            media_type.value, category=cache_category, base_key=cache_base_key
        ):
            for db_id in prev_library_items:
                if db_id not in cur_db_ids:
                    try:
                        item = await controller.get_library_item(db_id)
                    except MediaNotFoundError:
                        # edge case: the item is already removed
                        continue
                    remaining_providers = {
                        x.provider_domain
                        for x in item.provider_mappings
                        if x.provider_domain != self.domain
                    }
                    if remaining_providers:
                        continue
                    # this item is removed from the provider's library
                    # and we have no other providers attached to it
                    # it is safe to remove it from the MA library too
                    # note that we do not remove item's recursively on purpose
                    try:
                        await controller.remove_item_from_library(db_id, recursive=False)
                    except MusicAssistantError as err:
                        # this is probably because the item still has dependents
                        self.logger.warning(
                            "Error removing item %s from library: %s", db_id, str(err)
                        )
                        # just un-favorite the item if we can't remove it
                        await controller.set_favorite(db_id, False)
                    await asyncio.sleep(0)  # yield to eventloop

        await self.mass.cache.set(
            media_type.value,
            list(cur_db_ids),
            category=cache_category,
            base_key=cache_base_key,
        )

    # DO NOT OVERRIDE BELOW

    def library_supported(self, media_type: MediaType) -> bool:
        """Return if Library is supported for given MediaType on this provider."""
        if media_type == MediaType.ARTIST:
            return ProviderFeature.LIBRARY_ARTISTS in self.supported_features
        if media_type == MediaType.ALBUM:
            return ProviderFeature.LIBRARY_ALBUMS in self.supported_features
        if media_type == MediaType.TRACK:
            return ProviderFeature.LIBRARY_TRACKS in self.supported_features
        if media_type == MediaType.PLAYLIST:
            return ProviderFeature.LIBRARY_PLAYLISTS in self.supported_features
        if media_type == MediaType.RADIO:
            return ProviderFeature.LIBRARY_RADIOS in self.supported_features
        if media_type == MediaType.AUDIOBOOK:
            return ProviderFeature.LIBRARY_AUDIOBOOKS in self.supported_features
        if media_type == MediaType.PODCAST:
            return ProviderFeature.LIBRARY_PODCASTS in self.supported_features
        return False

    def library_edit_supported(self, media_type: MediaType) -> bool:
        """Return if Library add/remove is supported for given MediaType on this provider."""
        if media_type == MediaType.ARTIST:
            return ProviderFeature.LIBRARY_ARTISTS_EDIT in self.supported_features
        if media_type == MediaType.ALBUM:
            return ProviderFeature.LIBRARY_ALBUMS_EDIT in self.supported_features
        if media_type == MediaType.TRACK:
            return ProviderFeature.LIBRARY_TRACKS_EDIT in self.supported_features
        if media_type == MediaType.PLAYLIST:
            return ProviderFeature.LIBRARY_PLAYLISTS_EDIT in self.supported_features
        if media_type == MediaType.RADIO:
            return ProviderFeature.LIBRARY_RADIOS_EDIT in self.supported_features
        if media_type == MediaType.AUDIOBOOK:
            return ProviderFeature.LIBRARY_AUDIOBOOKS_EDIT in self.supported_features
        return False

    def _get_library_gen(self, media_type: MediaType) -> AsyncGenerator[MediaItemType, None]:
        """Return library generator for given media_type."""
        if media_type == MediaType.ARTIST:
            return self.get_library_artists()
        if media_type == MediaType.ALBUM:
            return self.get_library_albums()
        if media_type == MediaType.TRACK:
            return self.get_library_tracks()
        if media_type == MediaType.PLAYLIST:
            return self.get_library_playlists()
        if media_type == MediaType.RADIO:
            return self.get_library_radios()
        if media_type == MediaType.AUDIOBOOK:
            return self.get_library_audiobooks()
        if media_type == MediaType.PODCAST:
            return self.get_library_podcasts()
        raise NotImplementedError
