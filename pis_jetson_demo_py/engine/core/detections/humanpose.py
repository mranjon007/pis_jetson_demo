from __future__ import annotations

import copy
from typing import Dict, List, Tuple, Union

import numpy as np

from .base import DetectionItem, PassengerType


def _check_human_pose_attributes(joint_items: List[np.ndarray]) -> None:
    assert len(joint_items.shape) == 2 and joint_items.shape[1] == 3
    for joint in joint_items:
        pos_x, pos_y, conf = joint
        # assert 0 <= conf <= 1 or conf == -1  # Nondetected joint
        if not (0 <= conf <= 1 or conf == -1):
            conf = -1


def _parse_human_pose_joints(
    joint_items: List[np.ndarray],
) -> Dict[int, Tuple[int, int, float]]:
    all_joints = {}
    for joint_idx, joint in enumerate(joint_items[5:]):
        pos_x, pos_y, conf = joint
        all_joints[joint_idx] = (int(pos_x), int(pos_y), conf)

    return all_joints


class HumanPoseDetectionItem(DetectionItem):
    def __init__(
        self,
        bbox: np.ndarray,
        conf: float,
        class_id: float,
        human_pose: np.ndarray,
        passenger_type: PassengerType = PassengerType.NOT_SET,
    ):
        super(HumanPoseDetectionItem, self).__init__(
            bbox=bbox, conf=conf, class_id=class_id
        )
        _check_human_pose_attributes(human_pose)
        self.num_joints: int = human_pose.shape[0]
        self.joints: Dict[int, Tuple[int, int, float]] = _parse_human_pose_joints(
            human_pose
        )
        self.passenger_type = passenger_type

    def __str__(self) -> str:
        attributes = " ".join(
            [
                f"{key}={getattr(self, key)}"
                for key in [
                    "tracker_id",
                    "tracker_metrics",
                    "joints",
                    "passenger_type",
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
            copy.deepcopy(self.joints),
        ]

    def get_joints(self) -> Dict[int, Tuple[int, int, float]]:
        return copy.deepcopy(self.joints)
