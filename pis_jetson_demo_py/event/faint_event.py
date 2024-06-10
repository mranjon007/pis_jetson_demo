from enum import Enum
from typing import Any, Dict, List, Tuple

import numpy as np
from loguru import logger

from engine.core.detections import DetectionItem
from engine.core.detections.classes import ClassNamesManager
from engine.core.groupping import (
    HeadPoseDetectionItem,
    HumanPoseDetectionItem,
    PassengerType,
)

from .base import BaseEvent, BaseEventCallbackProcessor


class FaintType(Enum):
    DROP_HEAD = 1
    DROP_UPPER_BODY = 2


class FaintEvent(BaseEvent):
    def __init__(
        self,
        ptype: PassengerType,
        det_person: HumanPoseDetectionItem,
        ftype: FaintType,
    ):
        self.ptype: PassengerType = ptype
        self.det_person: HumanPoseDetectionItem = det_person
        self.ftype: FaintType = ftype


def argmax(items: List[Any]) -> int:
    if not items:
        return -1
    idx, item = next(
        iter(sorted(enumerate(items), key=lambda val: val[1], reverse=True))
    )
    return idx


def argmin(items: List[Any]) -> int:
    if not items:
        return -1
    idx, item = next(
        iter(sorted(enumerate(items), key=lambda val: val[1], reverse=False))
    )
    return idx


def find_person_from_head(
    head: HeadPoseDetectionItem, candidates: List[DetectionItem]
) -> HumanPoseDetectionItem:
    if not candidates:
        return None
    ious = [person_det.iou(head) for person_det in candidates]
    return candidates[argmax(ious)]


def largest_volume(detections: List[DetectionItem]) -> DetectionItem:
    volumes = []
    for det in detections:
        bbox = det.bbox()
        width, height = bbox[2:] - bbox[:2]
        volumes.append(width * height)

    if not volumes:
        return None
    return detections[argmax(volumes)]


def filter_face_joints(
    joints: Dict[int, Tuple[int, int, float]]
) -> Dict[int, Tuple[int, int, float]]:
    face_joint_idx = [0, 1, 2, 3, 4]
    return {key: value for key, value in joints.items() if key in face_joint_idx}


def filter_shoulder_joints(
    joints: Dict[int, Tuple[int, int, float]]
) -> Dict[int, Tuple[int, int, float]]:
    shoulder_joint_idx = [5, 6]
    return {key: value for key, value in joints.items() if key in shoulder_joint_idx}


def filter_nose_joint(
    joints: Dict[int, Tuple[int, int, float]]
) -> Tuple[int, int, float]:
    nose_joint_idx = 0
    if nose_joint_idx not in joints.keys():
        return None
    return joints[nose_joint_idx]


class OneshotCalibrator:
    def __init__(self, window: int = 5):
        self.window: int = window
        self.values: List[float] = []
        self.ttr = 10
        self.ttr_counter = 0

    def update(self, pose_values) -> None:
        if pose_values is None:
            self.ttr_counter += 1
            if self.ttr_counter >= self.ttr:
                self.values.clear()
            return

        self.ttr_counter = 0
        if len(self.values) <= self.window:
            self.values.append(pose_values)

    def mean(self) -> float:
        return 0 if not self.values else np.mean(self.values)


class FaintEventProcessor(BaseEventCallbackProcessor):
    LOG_HEADER: str = "[Faint] "

    def __init__(
        self,
        ttc: int,
        tte: int,
        ptype: PassengerType,
        src_dims: Tuple[int, int],
        face_shoulder_thresh: float,
        shoulder_collapse_thresh: float,
        nose_mean_thresh: float,
    ):
        """Processes faint event.

        Check the location of face keypoints (eyes, noses, ears) from HumanPose results
        and check the vertical location of them to detect faint event.

        1. Calibrate formal driver or passenger's shoulder position (mean position)
        2. Check-
          (1) the difference of shoulder position
          (2) the difference between avg. shoulder position and current face keypoint(spec. nose) position
        3. Put them in event window

        # Also if we don't have face keypoints, Try to infer the state
        # using shoulder keypoint location (will be differ based on passenger type)

        # Finally, some videos have no face keypoints at all or no movement of shoulder,
        # these should be considered using face detection results. when No face is detected,
        # then there are a lot of chance that the person is in faint status.

        Args:
            ttc (int): Time To Clear (event unseen counts until clearing existing events)
            tte (int): Time To Emit (event seen counts for making alarm/event/report)
            ptype (PassengerType): Specific passenger type to be watched.
        """
        super(FaintEventProcessor, self).__init__()

        self.event_window = []

        self.ttc = ttc
        self.tte = tte
        self.ptype = ptype
        self.ttc_counter = 0
        self.src_height, self.src_width = src_dims
        self.face_shoulder_thresh = face_shoulder_thresh
        self.shoulder_collapse_thresh = shoulder_collapse_thresh
        self.nose_mean_thresh = nose_mean_thresh
        self.shoulder_calibrator = OneshotCalibrator(window=5)
        self.nose_calibrator = OneshotCalibrator(window=5)

    def update(self, all_detections: List[DetectionItem]):
        # Filter passenger and driver

        persons: List[HumanPoseDetectionItem] = [
            det
            for det in all_detections
            if (
                isinstance(det, HumanPoseDetectionItem)
                and det.passenger_type == self.ptype
                and det.tracker_id is not None
            )
        ]

        if not persons:
            self.ttc_counter += 1
        else:
            person = largest_volume(persons)
            if self.detect_criteria(person):
                logger.debug(
                    __class__.LOG_HEADER + "Detected event, resetting TTC counter"
                )
                self.ttc_counter = 0
            else:
                self.ttc_counter += 1

        # Remove all events when Time To Clear conditions met
        if self.ttc_counter >= self.ttc:
            self.event_window.clear()

        # Reduce only of we have plenty of events
        if len(self.event_window) > self.tte:
            self.event_window.pop(0)

        self.send_event()

    def detect_criteria(
        self,
        person: HumanPoseDetectionItem,
    ) -> bool:
        """Detects the criteria of event and save event to event_window.
        Do not directly emit event on this function (we need to store on window-basis)

        Args:
            head (HeadPoseDetectionItem): Target head for checking yaw, pitch, roll value
            person (HumanPoseDetectionItem): Target person to identify event

        Returns:
            bool: True if event occurs, or False. Mandatory for clearing event window.
        """
        joints: Dict[int, Tuple[int, int, float]] = person.get_joints()
        joints = {
            key: (x, y, conf) for (key, (x, y, conf)) in joints.items() if conf != -1
        }

        if joints is None:
            return False

        live_joints = sum([1 for (x, y, conf) in joints.values() if conf != -1])
        if live_joints == 0:
            return False

        shoulder_joints = filter_shoulder_joints(joints)
        face_joints = filter_face_joints(joints)
        nose_joint = filter_nose_joint(joints)

        if len(shoulder_joints) == 0:
            return False

        # Calibrate values
        shoulder_joint_y_values = [y for (x, y, conf) in shoulder_joints.values()]
        if shoulder_joint_y_values:
            current_shoulder_mean = (
                0 if not shoulder_joint_y_values else np.mean(shoulder_joint_y_values)
            )
        else:
            current_shoulder_mean = -1

        if len(shoulder_joint_y_values) == 2:
            # These one-shot calibrator will only accept first WINDOW items and drop others.
            self.shoulder_calibrator.update(current_shoulder_mean)
        else:
            self.shoulder_calibrator.update(None)

        if nose_joint is not None:
            _, nose_y_value, _ = nose_joint
            self.nose_calibrator.update(nose_y_value)
        else:
            self.nose_calibrator.update(None)

        # Check if face keypoints are dropping below center threshold
        face_joint_y_values = [y for (x, y, conf) in face_joints.values()]
        shoulder_mean = self.shoulder_calibrator.mean()

        # Criteria 1-1: Partial face keypoint, Any of face keypoints
        #             below shoulder mean value
        if (
            len(face_joint_y_values) <= 3  # formarly 5
            and (
                np.array(face_joint_y_values)
                > shoulder_mean + (self.src_height * self.face_shoulder_thresh)
            ).any()
        ) or not face_joint_y_values:
            logger.debug(f"{self.ptype}, Criteria 1-1 - LessFaceAndOneOf")
            self.event_window.append(
                FaintEvent(
                    ptype=self.ptype,
                    det_person=person,
                    ftype=FaintType.DROP_UPPER_BODY,
                )
            )
            return True

        # Criteria 1-2: One of face joint should above shoulder mean
        if (
            np.array(face_joint_y_values)
            > shoulder_mean - (self.src_height * self.face_shoulder_thresh)
        ).all():
            logger.debug(f"{self.ptype}, Criteria 1-2 - AllOf")
            self.event_window.append(
                FaintEvent(
                    ptype=self.ptype,
                    det_person=person,
                    ftype=FaintType.DROP_UPPER_BODY,
                )
            )
            return True

        # Criteria 2: Whole shoulder down
        if shoulder_joint_y_values and (
            current_shoulder_mean
            > shoulder_mean + self.src_height * self.shoulder_collapse_thresh
        ):
            logger.debug(f"{self.ptype}, Criteria 2 - ShoulderCollapse")
            self.event_window.append(
                FaintEvent(
                    ptype=self.ptype,
                    det_person=person,
                    ftype=FaintType.DROP_UPPER_BODY,
                )
            )
            return True

        # Criteria 3: Mean nose location
        nose_mean = self.nose_calibrator.mean()
        if nose_joint is not None:
            _, nose_y_value, _ = nose_joint
            if nose_y_value > nose_mean + self.src_height * self.nose_mean_thresh:
                logger.debug(f"{self.ptype}, Criteria 3 - MeanNoseThresh")
                self.event_window.append(
                    FaintEvent(
                        ptype=self.ptype,
                        det_person=person,
                        ftype=FaintType.DROP_HEAD,
                    )
                )
                return True

        return False

    def send_event(self) -> None:
        # Check the thresholds and counts of the events
        # whether this event message should be emitted.

        if len(self.event_window) >= self.tte:
            if not self.emit(self.event_window[-1]):
                logger.warning("Failed to emit event to one of handlers.")
            else:
                logger.debug("Successfully sent event")
