from typing import Tuple

import numpy as np

from .base import BaseTracker
from utils.metric import iou


class IoUTracker(BaseTracker):
    """IoU-based Tracker

    This tracker calculates IoU to track objects.
    """

    DEFAULT_TRACKER_PARAMS = {
        "threshold": 0.5,
        "ttm": 3,
        "ttl": 3,
    }

    def __init__(
        self,
        threshold: float,
        ttm: int,
        ttl: int,
        image_size: Tuple[int, int],
    ):
        """IoU-based Tracker

        Args:
            min_iou_thresh (float): Minimal baseline IoU for matching. Defaults to 0.5.
                                    (If both boxes IoU are 0.4, will be considered as non-matching.)
            match_period (int): Period for constructing new track. Defaults to 3.
            ttl (int): Time to live for unmatched bounding boxes. Defaults to 3.
        """
        super(IoUTracker, self).__init__(
            threshold=threshold,
            ttm=ttm,
            ttl=ttl,
            image_size=image_size,
            best_metric_type="max",
        )

    def calculate_metric(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        return iou(bbox1, bbox2)

    def get_tracker_type(self) -> str:
        return "IoUTracker"
