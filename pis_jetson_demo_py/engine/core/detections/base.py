from __future__ import annotations

from enum import Enum
from typing import List, Union
from utils.metric import iou, l2_distance, intersection

import numpy as np


class PassengerType(Enum):
    NOT_SET = 0
    PASSENGER = 1
    DRIVER = 2
    PASSENGER_BACKSEAT = 3


def _check_bbox_attributes(bbox: np.ndarray, conf: float, class_id: float) -> None:
    assert len(bbox.shape) == 1 and len(bbox) == 4
    xmin, ymin, xmax, ymax = bbox
    assert xmax - xmin >= 0 and ymax - ymin >= 0
    assert not np.all(bbox <= 1)
    assert 0 <= conf <= 1
    assert abs(class_id - int(class_id)) < 1e-9


class DetectionItem:
    """Base Object Detection item.

    Also contains attributes of tracker ID and metrics (e.g. tracked values)
    """

    @classmethod
    def derive_from(cls, detection_item: DetectionItem, **kwargs) -> cls:
        """Derive new class from base DetectionItem.
        For example, extend HeadPoseDetectionItem from existing DetectionItem

        Args:
            detection_item (DetectionItem): original DetectionItem to extend its attribute from.
            **kwargs (Dict[str, Any]): Per-item additional attributes. See below implementations.
              - `engine.core.detections.headpose.HeadPoseDetectionItem`: `head_angles: np.ndarray`
              - `engine.core.detections.humanpose.HumanPoseDetectionItem`: `human_pose: np.ndarray`
        """
        (
            xmin,
            ymin,
            xmax,
            ymax,
            conf,
            class_id,
            tracker_id,
            tracker_metrics,
            *_,
        ) = detection_item.serialize()

        item = cls(
            bbox=np.array([xmin, ymin, xmax, ymax]),
            conf=conf,
            class_id=class_id,
            **kwargs,
        )

        item.tracker_id = tracker_id
        item.tracker_metrics = tracker_metrics

        return item

    def __init__(self, bbox: np.ndarray, conf: float, class_id: float):
        _check_bbox_attributes(bbox, conf, class_id)
        xmin, ymin, xmax, ymax = bbox.tolist()
        self.xmin = int(xmin)
        self.ymin = int(ymin)
        self.xmax = int(xmax)
        self.ymax = int(ymax)
        self.conf = conf
        self.class_id = int(class_id)
        self.tracker_id = None
        self.tracker_metrics = None

    def __str__(self) -> str:
        attributes = " ".join(
            [f"{key}={getattr(self, key)}" for key in ["tracker_id", "tracker_metrics"]]
        )
        description = f"<{self.__class__.__name__} ({self.xmin}, {self.ymin}, {self.xmax}, {self.ymax}){' ' + attributes if attributes else ''}>"
        return description

    def __repr__(self) -> str:
        return self.__str__()

    def iou(self, other: DetectionItem) -> float:
        return iou(self.bbox(), other.bbox())

    def intersection(self, other: DetectionItem) -> float:
        return intersection(self.bbox(), other.bbox())

    def abs_distance(self, other: DetectionItem) -> float:
        return l2_distance(self.bbox(), other.bbox(), 1, 1)

    def serialize(self) -> List[Union[int, float, None]]:
        return [
            self.xmin,
            self.ymin,
            self.xmax,
            self.ymax,
            self.conf,
            self.class_id,
            self.tracker_id,
            self.tracker_metrics,
        ]

    def bbox(self) -> np.ndarray:
        return np.array([self.xmin, self.ymin, self.xmax, self.ymax], dtype=int)


def parse_od_predictions(
    pred_boxes: np.ndarray, pred_scores: np.ndarray, pred_class_id: np.ndarray
) -> List[DetectionItem]:
    all_detection_items = []
    for bbox, conf, class_id in zip(pred_boxes, pred_scores, pred_class_id):
        all_detection_items.append(
            DetectionItem(
                bbox=bbox,
                conf=conf.item(),
                class_id=class_id.item(),
            )
        )

    return all_detection_items
