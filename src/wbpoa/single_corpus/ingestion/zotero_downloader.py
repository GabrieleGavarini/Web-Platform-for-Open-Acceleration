from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol

import requests
from requests.adapters import HTTPAdapter
from structlog import get_logger
from urllib3.util.retry import Retry

from wbpoa.core.corpus import ZoteroSourceConfig
from wbpoa.single_corpus.settings import ZoteroSettings

log = get_logger()

_PAGE_SIZE = 100
_SYNC_MARKER = ".zotero-sync.json"


class ZoteroDownloadError(RuntimeError):
    """Raised when a configured Zotero collection cannot be synchronized."""


class _HttpResponse(Protocol):
    content: bytes

    def raise_for_status(self) -> None:
        """Raise when the response status is not successful."""

    def json(self) -> object:
        """Decode and return the JSON response body."""

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        """Yield response body chunks."""
        ...


class _HttpSession(Protocol):
    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout: float,
        params: Mapping[str, int | str] | None = None,
        stream: bool = False,
    ) -> _HttpResponse:
        """Send an HTTP GET request."""
        ...


class ZoteroDownloader:
    """Synchronize PDF attachments from one corpus-specific set of Zotero collections.

    Attributes:
        source: Zotero group and collection identifiers from the corpus manifest.
        settings: Shared authentication, retry, timeout, and pacing settings.
        output_dir: Corpus-local directory receiving downloaded PDFs.
    """

    def __init__(
        self,
        source: ZoteroSourceConfig,
        settings: ZoteroSettings,
        output_dir: Path,
        *,
        session: _HttpSession | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Initialize a Zotero collection synchronizer.

        Args:
            source: Group and collection identifiers from the corpus manifest.
            settings: Shared Zotero API configuration.
            output_dir: Corpus-local destination for the collections' PDFs.
            session: Optional HTTP session supplied by callers.
            sleep: Delay function used to pace file downloads.
        """
        self.source = source
        self.settings = settings
        self.output_dir = output_dir
        self._session = session if session is not None else self._build_session(settings.max_retries)
        self._sleep = sleep

    @staticmethod
    def _build_session(max_retries: int) -> requests.Session:
        retry = Retry(
            total=max_retries,
            connect=max_retries,
            read=max_retries,
            status=max_retries,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods={"GET"},
            backoff_factor=0.5,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session = requests.Session()
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def sync(self, *, refresh: bool = False) -> list[Path]:
        """Download every configured collection or reuse their complete local snapshot.

        Args:
            refresh: Whether to ignore a valid synchronization marker and fetch the collections again.

        Returns:
            Sorted paths of the synchronized PDF attachments.

        Raises:
            ZoteroDownloadError: If credentials are absent, API data is malformed, or synchronization fails.
        """
        cached = self._cached_paths()
        if cached is not None and not refresh:
            log.info("zotero_sync_skipped", output_dir=str(self.output_dir), files=len(cached))
            return cached

        log.info(
            "zotero_sync_started",
            group_id=self.source.group_id,
            collections=self.source.collection_keys,
            output_dir=str(self.output_dir),
            refresh=refresh,
        )

        api_key = self.settings.api_key
        if api_key is None:
            message = "ZOTERO_API_KEY is required to synchronize a configured collection"
            raise ZoteroDownloadError(message)

        headers = {"Zotero-API-Key": api_key.get_secret_value(), "Zotero-API-Version": "3"}
        try:
            items_by_key: dict[str, dict[str, object]] = {}
            for collection_key in self.source.collection_keys:
                log.info("zotero_collection_sync_started", collection=collection_key)
                for item in self._get_all_items(headers, collection_key):
                    key = item.get("key")
                    if isinstance(key, str):
                        items_by_key[key] = item
                log.info(
                    "zotero_collection_sync_completed",
                    collection=collection_key,
                    items=len(items_by_key),
                )
            paths = self._download_attachments(list(items_by_key.values()), headers)
            self._write_marker(paths)
        except (OSError, requests.RequestException, ValueError, KeyError, TypeError) as error:
            raise ZoteroDownloadError(
                f"Could not synchronize Zotero group {self.source.group_id}, collections {self.source.collection_keys}"
            ) from error
        log.info("zotero_sync_completed", files=len(paths), output_dir=str(self.output_dir))
        return paths

    def _get_all_items(self, headers: Mapping[str, str], collection_key: str) -> list[dict[str, object]]:
        items: list[dict[str, object]] = []
        start = 0
        while True:
            log.debug("zotero_collection_page_requested", collection=collection_key, start=start, limit=_PAGE_SIZE)
            response = self._session.get(
                self._collection_url(collection_key),
                headers=headers,
                params={"format": "json", "limit": _PAGE_SIZE, "start": start},
                timeout=self.settings.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                message = "Zotero collection response must be a list"
                raise TypeError(message)
            batch = [self._as_item(value) for value in payload]
            items.extend(batch)
            log.debug(
                "zotero_collection_page_received",
                collection=collection_key,
                start=start,
                items=len(batch),
                total=len(items),
            )
            if len(batch) < _PAGE_SIZE:
                return items
            start += _PAGE_SIZE

    def _download_attachments(
        self,
        items: list[dict[str, object]],
        headers: Mapping[str, str],
    ) -> list[Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        by_key = {key: item for item in items if (key := item.get("key")) and isinstance(key, str)}
        attachments = [item for item in items if self._is_pdf_attachment(item)]
        log.info("zotero_attachments_discovered", items=len(items), attachments=len(attachments))
        reserved_names: set[str] = set()
        paths: list[Path] = []
        for index, attachment in enumerate(attachments):
            key = self._required_string(attachment, "key")
            data = self._item_data(attachment)
            parent_key = data.get("parentItem")
            parent = by_key.get(parent_key) if isinstance(parent_key, str) else None
            filename = self._attachment_filename(key, data, parent, reserved_names)
            target_path = self.output_dir / filename
            log.debug("zotero_attachment_download_started", key=key, filename=filename, index=index + 1)
            if self._download_file(key, target_path, headers):
                paths.append(target_path)
            if index + 1 < len(attachments):
                self._sleep(self.settings.rate_limit_seconds)
        return sorted(paths)

    def _download_file(self, key: str, target_path: Path, headers: Mapping[str, str]) -> bool:
        """Download a single attachment, skipping it if Zotero serves empty content.

        Args:
            key: Zotero attachment item key.
            target_path: Destination path for the downloaded file.
            headers: Zotero API authentication headers.

        Returns:
            Whether the attachment was downloaded successfully.
        """
        temporary_path: Path | None = None
        try:
            response = self._session.get(
                f"{self._base_url}/items/{key}/file",
                headers=headers,
                timeout=self.settings.timeout_seconds,
                stream=True,
            )
            response.raise_for_status()
            with NamedTemporaryFile(
                dir=self.output_dir, prefix=f".{target_path.name}.", suffix=".part", delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                byte_count = 0
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        temporary.write(chunk)
                        byte_count += len(chunk)
                temporary.flush()
                os.fsync(temporary.fileno())
        except Exception:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise
        if byte_count == 0:
            temporary_path.unlink(missing_ok=True)
            log.warning("zotero_attachment_empty_skipped", key=key, filename=target_path.name)
            return False
        temporary_path.replace(target_path)
        log.debug("zotero_attachment_download_completed", key=key, filename=target_path.name, bytes=byte_count)
        return True

    def _cached_paths(self) -> list[Path] | None:
        marker_path = self.output_dir / _SYNC_MARKER
        if not marker_path.is_file():
            existing_pdfs = sorted(
                path for path in self.output_dir.glob("*") if path.is_file() and path.suffix.lower() == ".pdf"
            )
            return existing_pdfs or None
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            valid_marker = (
                isinstance(marker, dict)
                and marker.get("group_id") == self.source.group_id
                and marker.get("collection_keys") == sorted(self.source.collection_keys)
            )
            filenames = marker.get("files") if valid_marker else None
            if not isinstance(filenames, list) or not all(isinstance(name, str) for name in filenames):
                return None
            paths = [self.output_dir / name for name in filenames]
            return sorted(paths) if all(path.is_file() for path in paths) else None
        except (OSError, json.JSONDecodeError):
            return None

    def _write_marker(self, paths: list[Path]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        marker_path = self.output_dir / _SYNC_MARKER
        payload = {
            "group_id": self.source.group_id,
            "collection_keys": sorted(self.source.collection_keys),
            "files": [path.name for path in sorted(paths)],
        }
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.output_dir,
            prefix=f".{_SYNC_MARKER}.",
            suffix=".part",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, indent=2)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(marker_path)

    @property
    def _base_url(self) -> str:
        return f"https://api.zotero.org/groups/{self.source.group_id}"

    def _collection_url(self, collection_key: str) -> str:
        return f"{self._base_url}/collections/{collection_key}/items"

    @staticmethod
    def _as_item(value: object) -> dict[str, object]:
        if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
            message = "Zotero item must be an object with string keys"
            raise TypeError(message)
        return value

    @staticmethod
    def _item_data(item: dict[str, object]) -> dict[str, object]:
        data = item.get("data")
        if not isinstance(data, dict) or not all(isinstance(key, str) for key in data):
            message = "Zotero item data must be an object with string keys"
            raise TypeError(message)
        return data

    @staticmethod
    def _required_string(item: dict[str, object], key: str) -> str:
        value = item.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Zotero item {key} must be a non-empty string")
        return value

    @classmethod
    def _is_pdf_attachment(cls, item: dict[str, object]) -> bool:
        try:
            data = cls._item_data(item)
        except TypeError:
            return False
        filename = data.get("filename")
        return (
            data.get("itemType") == "attachment"
            and data.get("linkMode") in {"imported_file", "imported_url"}
            and (
                data.get("contentType") == "application/pdf"
                or (isinstance(filename, str) and filename.lower().endswith(".pdf"))
            )
        )

    @classmethod
    def _attachment_filename(
        cls,
        key: str,
        data: dict[str, object],
        parent: dict[str, object] | None,
        reserved_names: set[str],
    ) -> str:
        label = cls._parent_label(parent)
        raw_filename = data.get("filename")
        fallback = raw_filename if isinstance(raw_filename, str) and raw_filename else f"{key}.pdf"
        filename = f"{label}.pdf" if label else cls._safe_name(fallback)
        if filename in reserved_names:
            filename = f"{Path(filename).stem} [{key}].pdf"
        reserved_names.add(filename)
        return filename

    @classmethod
    def _parent_label(cls, parent: dict[str, object] | None) -> str:
        if parent is None:
            return ""
        data = cls._item_data(parent)
        creators = data.get("creators")
        author = ""
        if isinstance(creators, list) and creators and isinstance(creators[0], dict):
            creator = creators[0]
            last_name = creator.get("lastName")
            creator_name = creator.get("name")
            author = last_name if isinstance(last_name, str) else creator_name if isinstance(creator_name, str) else ""
        date = data.get("date")
        title = data.get("title")
        year = date[:4] if isinstance(date, str) else ""
        title = title if isinstance(title, str) else ""
        return cls._safe_name(" - ".join(part for part in (author, year, title) if part))

    @staticmethod
    def _safe_name(text: str, max_length: int = 150) -> str:
        sanitized = re.sub(r'[<>:"/\\|?*\n\r\t]', "-", text)
        sanitized = re.sub(r"\s+", " ", sanitized).strip(" .")
        return sanitized[:max_length] or "untitled"
