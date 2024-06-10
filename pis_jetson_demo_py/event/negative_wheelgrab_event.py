from typing import Any, List, Tuple, Dict

from loguru import logger
from shapely.geometry import Point, Polygon

from engine.core.detections import DetectionItem
from engine.core.detections.classes import ClassNamesManager
from engine.core.groupping import (
    HeadPoseDetectionItem,
    HumanPoseDetectionItem,
    PassengerType,
)

from .base import BaseEvent, BaseEventCallbackProcessor


class NegativeWheelGrabEvent(BaseEvent):
    def __init__(
        self,
        ptype: PassengerType,
        det_person: HumanPoseDetectionItem,
    ):
        self.ptype: PassengerType = ptype
        self.det_person: HumanPoseDetectionItem = det_person


class WheelRoiManager:
    def __init__(self, roi: List[Tuple[float, float]]):
        self.roi = Polygon(roi)

    def check(self, y: float, x: float):
        return self.roi.contains(Point((x, y)))


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


def filter_all_arm_joints(
    joints: Dict[int, Tuple[int, int, float]]
) -> Dict[int, Tuple[int, int, float]]:
    arm_joint_idx = [7, 8, 9, 10]
    return {key: value for key, value in joints.items() if key in arm_joint_idx}


def filter_right_wrist_joint(
    joints: Dict[int, Tuple[int, int, float]]
) -> Tuple[int, int, float]:
    right_wrist_joint_idx = 10
    if right_wrist_joint_idx not in joints.keys():
        return None
    return joints[right_wrist_joint_idx]


WHEEL_ROI_RHS = [
    (0.6696428571428571, 1.0000000000000000),
    (0.6964285714285714, 0.8351648351648352),
    (1.0000000000000000, 0.7802197802197802),
    (1.0000000000000000, 1.0000000000000000),
]


class NegativeWheelGrabEventProcessor(BaseEventCallbackProcessor):
    LOG_HEADER: str = "[NegativeWheelGrab] "

    def __init__(
        self,
        ttc: int,
        tte: int,
        ptype: PassengerType,
        src_dims: Tuple[int, int],
    ):
        """Processes negative whell grab event.

        Check the location of both wrist and elbow to filter out non-wheelgrab events.

        Args:
            ttc (int): Time To Clear (event unseen counts until clearing existing events)
            tte (int): Time To Emit (event seen counts for making alarm/event/report)
            ptype (PassengerType): Specific passenger type to be watched.
        """
        super(NegativeWheelGrabEventProcessor, self).__init__()

        self.event_window = []

        self.ttc = ttc
        self.tte = tte
        self.ptype = ptype
        self.ttc_counter = 0
        self.rhs_roi = WheelRoiManager(WHEEL_ROI_RHS)
        self.src_height, self.src_width = src_dims

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

        all_arm_joints = filter_all_arm_joints(joints)
        right_wrist_joint = filter_right_wrist_joint(joints)

        # Criteria: Right arm joints are not detected
        if len(all_arm_joints) <= 2:
            return False

        if right_wrist_joint is None:
            return False

        # Criteria: Right wrist handling the wheel
        right_wrist_x_value, right_wrist_y_value, _ = right_wrist_joint
        if self.rhs_roi.check(
            right_wrist_y_value / self.src_height,
            right_wrist_x_value / self.src_width,
        ):
            return False

        self.event_window.append(
            NegativeWheelGrabEvent(
                ptype=self.ptype,
                det_person=person,
            )
        )
        return True

    def send_event(self) -> None:
        # Check the thresholds and counts of the events
        # whether this event message should be emitted.

        if len(self.event_window) >= self.tte:
            if not self.emit(self.event_window[-1]):
                logger.warning("Failed to emit event to one of handlers.")
            else:
                logger.debug("Successfully sent event")
