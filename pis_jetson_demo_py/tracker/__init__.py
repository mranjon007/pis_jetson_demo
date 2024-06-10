from .base import BaseTracker
from .iou import IoUTracker
from .l2 import L2Tracker
from .dummy import DummyTracker


def get_tracker_engine(engine_type: str = "iou", **tracker_kwargs) -> BaseTracker:
    """Get tracker engine for given type.

    Args:
        engine_type (str, optional): Tracker engine type. Defaults to "iou".

    Returns:
        BaseTracker: Tracker engine
    """
    assert engine_type in ["iou", "l2", "none"]

    if engine_type == "iou":
        tracker_params = {}
        tracker_params.update(IoUTracker.DEFAULT_TRACKER_PARAMS)
        tracker_params.update(tracker_kwargs)
        return IoUTracker(**tracker_params)
    elif engine_type == "l2":
        tracker_params = {}
        tracker_params.update(L2Tracker.DEFAULT_TRACKER_PARAMS)
        tracker_params.update(tracker_kwargs)
        return L2Tracker(**tracker_params)
    elif engine_type == "none":
        return DummyTracker()
