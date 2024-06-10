from typing import Any, List, Tuple

from loguru import logger

from engine.core.detections import DetectionItem
from engine.core.detections.classes import ClassNamesManager
from engine.core.groupping import (
    HeadPoseDetectionItem,
    HumanPoseDetectionItem,
    PassengerType,
)

from .base import BaseEvent, BaseEventCallbackProcessor


class DistractionEvent(BaseEvent):
    def __init__(
        self,
        ptype: PassengerType,
        det_head: HeadPoseDetectionItem,
        det_person: HumanPoseDetectionItem,
        state_yaw: bool,
        state_pitch: bool,
        state_front: bool,
    ):
        self.ptype: PassengerType = ptype
        self.det_head: HeadPoseDetectionItem = det_head
        self.det_person: HumanPoseDetectionItem = det_person
        self.state_yaw: bool = state_yaw
        self.state_pitch: bool = state_pitch
        self.state_front: bool = state_front


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


def get_distraction_event(yaw_calib, pitch_calib, yaw_warn, pitch_warn):
    yaw_thresh = 0.9
    pitch_thresh = 0.9

    state = []
    front_state = False
    yaw_state = False
    pitch_state = False

    # calib = differnece
    # = abs(head_value - mean_value)
    if yaw_calib > yaw_warn:
        yaw_state = True

    else:
        yaw_state = False

    if pitch_calib > pitch_warn:
        pitch_state = True

    else:
        pitch_state = False

    if (yaw_calib < yaw_warn * yaw_thresh) and (
        pitch_calib < pitch_warn * pitch_thresh
    ):
        front_state = True

    else:
        front_state = False

    state = [front_state, yaw_state, pitch_state]

    return state


class DistractionEventProcessor(BaseEventCallbackProcessor):
    LOG_HEADER: str = "[Distracted] "

    def __init__(
        self,
        ttc: int,
        tte: int,
        ptype: PassengerType,
    ):
        super(DistractionEventProcessor, self).__init__()

        self.event_window = []

        self.ttc = ttc
        self.tte = tte
        self.ptype = ptype
        self.ttc_counter = 0

        self.phone_class_ids = ClassNamesManager.get_instance().get_phone_class_ids()

    def update(self, all_detections: List[DetectionItem]):
        # Filter passenger and driver

        heads: List[HeadPoseDetectionItem] = [
            det
            for det in all_detections
            if (
                isinstance(det, HeadPoseDetectionItem)
                and det.passenger_type == self.ptype
                and det.tracker_id is not None
            )
        ]
        persons: List[HumanPoseDetectionItem] = [
            det
            for det in all_detections
            if (
                isinstance(det, HumanPoseDetectionItem)
                and det.passenger_type == self.ptype
                and det.tracker_id is not None
            )
        ]

        if not (heads and persons):
            self.ttc_counter += 1
        else:
            head = largest_volume(heads)
            person = find_person_from_head(head, persons)
            if self.detect_criteria(head, person):
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
        head: HeadPoseDetectionItem,
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
        distraction_event = head.distract_events()
        if distraction_event is None:
            return False

        front_state, yaw_state, pitch_state = distraction_event

        # Only create event when direction is not front (distracted)
        if not front_state:
            self.event_window.append(
                DistractionEvent(
                    ptype=self.ptype,
                    det_head=head,
                    det_person=person,
                    state_front=front_state,
                    state_yaw=yaw_state,
                    state_pitch=pitch_state,
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
