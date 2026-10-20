"""High-performance video reader using OpenCV with caching."""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Self

import cv2
import numpy as np


class VideoReader:
    """Fast, cached video frame reader wrapping OpenCV."""

    def __init__(self, video_path: str | Path, cache_size: int = 64) -> None:
        self.path = Path(video_path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {self.path}")

        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {self.path}")

        self.width: int = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height: int = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps: float = float(self._cap.get(cv2.CAP_PROP_FPS))
        if self.fps <= 0.0 or np.isnan(self.fps):
            self.fps = 30.0  # Reasonable default

        self.total_frames: int = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.total_frames <= 0:
            self.total_frames = 1

        self.duration: float = self.total_frames / self.fps
        self._current_pos: int = -1
        self._cache: collections.OrderedDict[int, np.ndarray] = collections.OrderedDict()
        self._cache_size: int = max(4, cache_size)

    def get_frame(self, frame_idx: int, to_rgb: bool = True) -> np.ndarray | None:
        """Retrieves the frame at the specified 0-based index.

        Args:
            frame_idx: 0-based integer frame index.
            to_rgb: If True, returns image in RGB format; otherwise BGR.

        Returns:
            Image array (uint8) or None if frame could not be read.
        """
        if frame_idx < 0 or frame_idx >= self.total_frames:
            return None

        # Check cache
        if frame_idx in self._cache:
            frame_bgr = self._cache[frame_idx]
            self._cache.move_to_end(frame_idx)
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) if to_rgb else frame_bgr.copy()

        # Seek if not at sequential position
        if self._current_pos != frame_idx:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame_bgr = self._cap.read()
        if not ret or frame_bgr is None:
            return None

        self._current_pos = frame_idx + 1

        # Store in cache
        self._cache[frame_idx] = frame_bgr
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) if to_rgb else frame_bgr.copy()

    def get_frame_gray(self, frame_idx: int) -> np.ndarray | None:
        """Retrieves the frame as a grayscale image."""
        frame_bgr = self.get_frame(frame_idx, to_rgb=False)
        if frame_bgr is None:
            return None
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    def close(self) -> None:
        """Releases the underlying video capture device."""
        if hasattr(self, "_cap") and self._cap is not None:
            self._cap.release()
        self._cache.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
