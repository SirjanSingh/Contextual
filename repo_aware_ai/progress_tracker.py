"""Progress tracking for upload and indexing operations."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class UploadProgress:
    """Tracks progress of an upload/indexing operation."""

    upload_id: str
    stage: str = "uploading"  # uploading | scanning | parsing | embedding | indexing | complete | error
    progress: float = 0.0  # 0-100
    total_files: int = 0
    files_processed: int = 0
    current_file: str = ""
    chunks_created: int = 0
    errors: List[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    repo_path: str = ""

    def update(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def calculate_eta(self) -> float:
        """Estimate remaining seconds."""
        elapsed = time.time() - self.start_time
        if self.progress <= 0:
            return -1
        total_estimated = elapsed / (self.progress / 100.0)
        return max(0, total_estimated - elapsed)

    def to_dict(self) -> dict:
        return {
            "upload_id": self.upload_id,
            "stage": self.stage,
            "progress": round(self.progress, 1),
            "total_files": self.total_files,
            "files_processed": self.files_processed,
            "current_file": self.current_file,
            "chunks_created": self.chunks_created,
            "errors": self.errors,
            "eta_seconds": round(self.calculate_eta(), 1),
            "elapsed_seconds": round(time.time() - self.start_time, 1),
        }


# Global progress store
_progress_store: Dict[str, UploadProgress] = {}


def create_progress(total_files: int = 0) -> UploadProgress:
    """Create a new progress tracker."""
    uid = str(uuid.uuid4())[:8]
    p = UploadProgress(upload_id=uid, total_files=total_files)
    _progress_store[uid] = p
    return p


def get_progress(upload_id: str) -> Optional[UploadProgress]:
    """Get progress by upload ID."""
    return _progress_store.get(upload_id)


def remove_progress(upload_id: str) -> None:
    """Remove a completed progress tracker."""
    _progress_store.pop(upload_id, None)
