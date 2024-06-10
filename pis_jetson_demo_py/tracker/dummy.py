from typing import Tuple

import numpy as np

from .base import BaseTracker


class DummyTracker(BaseTracker):
    """Dummy Tracker

    This tracker does basically nothing.
    """

    def __init__(self):
        super(DummyTracker, self).__init__(
            threshold=0.0,
            ttm=1e9,
            ttl=0,
            image_size=[None, None],
            best_metric_type="min",
        )

    def calculate_metric(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        # Return highest value possible to prevent matching
        return 1e9

    def get_tracker_type(self) -> str:
        return "DummyTracker"
