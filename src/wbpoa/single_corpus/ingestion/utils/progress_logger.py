from __future__ import annotations

import time


class ProgressLogger:
    """
    Tracks elapsed and estimated-remaining time across a fixed-size loop, for progress logging.

    Attributes:
        total: Total number of items the loop will process.
    """

    def __init__(self, total: int) -> None:
        """
        Initialize the tracker and start its internal clock.

        Args:
            total: Total number of items the loop will process.
        """
        self.total = total
        self._start = time.perf_counter()

    def snapshot(self, current_document: int, completed_documents: int) -> dict[str, int | float | None]:
        """
        Compute progress fields for the current document.

        Args:
            current_document: 1-indexed document currently being processed.
            completed_documents: Number of documents completed so far.

        Returns:
            dict[str, int | float | None]: Current/total counts, elapsed time, and estimated
            remaining duration. The estimate is ``None`` until a document has completed.
        """
        elapsed = time.perf_counter() - self._start
        estimated_remaining = (
            round((elapsed / completed_documents) * (self.total - completed_documents), 3)
            if completed_documents > 0
            else None
        )
        return {
            "current_document": current_document,
            "total_documents": self.total,
            "elapsed_seconds": round(elapsed, 3),
            "estimated_remaining_seconds": estimated_remaining,
        }

    def record(self, n: int) -> dict[str, int | float | None]:
        """Return a completion snapshot for the n-th document."""
        return self.snapshot(current_document=n, completed_documents=n)
