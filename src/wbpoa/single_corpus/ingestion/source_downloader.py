from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol

import requests
from pydantic import HttpUrl
from requests.adapters import HTTPAdapter
from structlog import get_logger
from urllib3.util.retry import Retry

log = get_logger()


class SourceDownloadError(RuntimeError):
    """Raised when a legislation source cannot be downloaded safely."""


class _HttpResponse(Protocol):
    def raise_for_status(self) -> None:
        """Raise when the response status is not successful."""
        ...

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        """Yield response body chunks."""
        ...


class _HttpSession(Protocol):
    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        stream: bool,
        timeout: float,
    ) -> _HttpResponse:
        """Send an HTTP GET request."""
        ...


class SourceDownloader:
    """Download a legislation source atomically while preserving a valid cache.

    Attributes:
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        *,
        timeout: float = 60.0,
        max_retries: int = 3,
        session: _HttpSession | None = None,
    ) -> None:
        """Initialize the downloader.

        Args:
            timeout: Per-request timeout in seconds.
            max_retries: Retry count for transient connection, server, and rate-limit failures.
            session: Optional HTTP session supplied by callers.

        Raises:
            ValueError: If timeout is not positive or max_retries is negative.
        """
        if timeout <= 0:
            message = "timeout must be positive"
            raise ValueError(message)
        if max_retries < 0:
            message = "max_retries must be non-negative"
            raise ValueError(message)
        self.timeout = timeout
        self._session = session if session is not None else self._build_session(max_retries)

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

    def download(self, url: HttpUrl | str, target_path: Path, *, refresh: bool = False) -> Path:
        """Download ``url`` to ``target_path`` unless a cached source may be reused.

        Args:
            url: HTTP(S) source URL.
            target_path: Exact destination filename.
            refresh: Whether to replace an existing source with a fresh download.

        Returns:
            The existing or newly downloaded source path.

        Raises:
            SourceDownloadError: If the request fails or returns an empty body.
        """
        if target_path.is_file() and not refresh:
            log.info("source_download_skipped", path=str(target_path), reason="already_exists")
            return target_path

        target_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            response = self._session.get(
                str(url),
                headers={
                    "Accept": "application/xhtml+xml",
                    "Accept-Language": "eng",
                },
                stream=True,
                timeout=self.timeout,
            )
            response.raise_for_status()
            with NamedTemporaryFile(
                dir=target_path.parent, prefix=f".{target_path.name}.", suffix=".part", delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                byte_count = 0
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        temporary.write(chunk)
                        byte_count += len(chunk)
                temporary.flush()
                os.fsync(temporary.fileno())
            self._require_content(byte_count, url)
            temporary_path.replace(target_path)
        except (OSError, requests.RequestException, SourceDownloadError) as error:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            if isinstance(error, SourceDownloadError):
                raise
            raise SourceDownloadError(f"Could not download legislation source from {url}") from error

        log.info("source_downloaded", url=str(url), path=str(target_path), bytes=byte_count)
        return target_path

    @staticmethod
    def _require_content(byte_count: int, url: HttpUrl | str) -> None:
        if byte_count == 0:
            raise SourceDownloadError(f"Downloaded source is empty: {url}")
