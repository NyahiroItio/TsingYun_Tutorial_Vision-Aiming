"""Per-frame target selection with digit-lock persistence.

Keeps the gun locked on one MNIST board across frames instead of
jumping between detections every frame.
"""

from __future__ import annotations

from detector import Detection


class TargetSelector:
    """Selects which detection to track across frames.

    Strategy: "digit-lock + digit-priority acquisition"

    1. When idle, keep detections with confidence >= threshold and class_id >= 0.
    2. Acquire the detection with largest class_id; tie-break by confidence.
    2. While tracking digit N, only match detections with class_id == N.
    3. If multiple detections share the same digit, pick highest confidence.
    4. After N consecutive frames with no match, release the lock and go idle.
    """

    def __init__(self, lost_threshold: int = 5, min_confidence: float = 0.5) -> None:
        self._lost_threshold = lost_threshold
        self._min_confidence = min_confidence
        self._tracking_digit: int | None = None
        self._lost_frames: int = 0

    @property
    def tracking_digit(self) -> int | None:
        return self._tracking_digit

    @property
    def is_tracking(self) -> bool:
        return self._tracking_digit is not None

    def select(self, detections: list[Detection]) -> Detection | None:
        if not self.is_tracking:
            return self._acquire(detections)
        return self._match(detections)

    def _acquire(self, detections: list[Detection]) -> Detection | None:
        if not detections:
            return None
        candidates = [
            d
            for d in detections
            if d.class_id >= 0 and d.confidence >= self._min_confidence
        ]
        if not candidates:
            return None
        # TODO: Pick largest digit, tie-break by confidence
        best = max(candidates, key=lambda d: (d.class_id, d.confidence))
        self._tracking_digit = best.class_id
        self._lost_frames = 0
        return best

    def _match(self, detections: list[Detection]) -> Detection | None:
        matches = [d for d in detections if d.class_id == self._tracking_digit]
        if not matches:
            self._lost_frames += 1
            if self._lost_frames > self._lost_threshold:
                self.reset()
            return None
        self._lost_frames = 0
        return max(matches, key=lambda d: d.confidence)

    def reset(self) -> None:
        self._tracking_digit = None
        self._lost_frames = 0
