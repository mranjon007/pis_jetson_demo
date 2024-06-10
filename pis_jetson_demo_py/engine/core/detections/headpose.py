from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np

from .base import DetectionItem, PassengerType


def _check_angle_attributes(angles: np.ndarray) -> None:
    assert len(angles.shape) == 1 and len(angles) == 3
    pitch, yaw, roll = angles
    assert -90 <= pitch <= 90, f"Invalid angle: {pitch}"
    assert -90 <= yaw <= 90, f"Invalid angle: {yaw}"
    assert -90 <= roll <= 90, f"Invalid angle: {roll}"


class HeadPoseDetectionItem(DetectionItem):
    def __init__(
        self,
        bbox: np.ndarray,
        conf: float,
        class_id: float,
        head_angles: np.ndarray,
        passenger_type: PassengerType = PassengerType.NOT_SET,
        events: np.ndarray = None,
    ):
        super(HeadPoseDetectionItem, self).__init__(
            bbox=bbox, conf=conf, class_id=class_id
        )
        _check_angle_attributes(head_angles)
        self.pitch, self.yaw, self.roll = head_angles
        self.passenger_type = passenger_type
        self.events = events

    def __str__(self) -> str:
        attributes = " ".join(
            [
                f"{key}={getattr(self, key)}"
                for key in [
                    "tracker_id",
                    "tracker_metrics",
                    "pitch",
                    "yaw",
                    "roll",
                    "passenger_type",
                    "events",
                ]
            ]
        )
        description = f"<{self.__class__.__name__} ({self.xmin}, {self.ymin}, {self.xmax}, {self.ymax}){' ' + attributes if attributes else ''}>"
        return description

    def __repr__(self) -> str:
        return self.__str__()

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
            self.passenger_type,
            self.pitch,
            self.yaw,
            self.roll,
        ]

    def rotation_metrics(self) -> Tuple[float, float, float]:
        return (self.pitch, self.yaw, self.roll)

    def distract_events(self) -> Tuple[bool, bool, bool]:
        if self.events is None:
            return None

        front_state = self.events[0]
        yaw_state = self.events[1]
        pitch_state = self.events[2]
        return front_state, yaw_state, pitch_state
