from typing import Tuple

import numpy as np

from .base import BaseTracker
from utils.metric import l2_distance


class L2Tracker(BaseTracker):
    """IoU-based Tracker

    This tracker calculates L2 distance to track objects.
    """

    DEFAULT_TRACKER_PARAMS = {
        "threshold": 0.4,
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
        """L2 distance-based Tracker

        Args:
            max_distance (float): L2 distance threshold for matching. Defaults to 0.4.
                                  (If tracked boxes center are farther than about 2/5 of screen
                                   will be considered as non-matching.)
            match_period (int): Period for constructing new track. Defaults to 3.
            ttl (int): Time to live for unmatched bounding boxes. Defaults to 3.
            image_size (Tuple[int, int]): Image size for normalizing values.
        """
        super(L2Tracker, self).__init__(
            threshold=threshold,
            ttm=ttm,
            ttl=ttl,
            image_size=image_size,
            best_metric_type="min",
        )

    def calculate_metric(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        img_width, img_height = self.image_size[0], self.image_size[1]
        return l2_distance(bbox1, bbox2, img_width, img_height)

    def get_tracker_type(self) -> str:
        return "L2Tracker"
