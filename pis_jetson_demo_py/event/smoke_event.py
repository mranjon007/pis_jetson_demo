from typing import Any, List

from loguru import logger

from engine.core.detections import DetectionItem
from engine.core.detections.classes import ClassNamesManager
from engine.core.groupping import (
    HeadPoseDetectionItem,
    HumanPoseDetectionItem,
    PassengerType,
)

from .base import BaseEvent, BaseEventCallbackProcessor


class SmokeEvent(BaseEvent):
    def __init__(
        self,
        ptype: PassengerType,
        det_phone: DetectionItem,
        det_head: HeadPoseDetectionItem,
        det_person: HumanPoseDetectionItem,
    ):
        self.ptype: PassengerType = ptype
        self.det_phone: DetectionItem = det_phone
        self.det_head: HeadPoseDetectionItem = det_head
        self.det_person: HumanPoseDetectionItem = det_person


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


class SmokeEventProcessor(BaseEventCallbackProcessor):
    LOG_HEADER: str = "[Smoke] "

    def __init__(
        self,
        ttc: int,
        tte: int,
        ptype: PassengerType,
        center_abs_distance_thresh: float = 100,
    ):
        """Processes smoking event.
        Calculate distance between cigarette and person's head center
        to recognize that the specific person is in call state.

        Args:
            ttc (int): Time To Clear (event unseen counts until clearing existing events)
            tte (int): Time To Emit (event seen counts for making alarm/event/report)
            ptype (PassengerType): Specific passenger type to be watched.
            center_abs_distance_thresh (float, optional): L2 absolute distance threshold between head center and phone. Defaults to 100.
        """
        super(SmokeEventProcessor, self).__init__()

        self.event_window = []
        self.ttc = ttc
        self.tte = tte
        self.ptype = ptype
        self.distance_thresh = center_abs_distance_thresh
        self.ttc_counter = 0

        self.phone_class_ids = ClassNamesManager.get_instance().get_smoke_class_ids()

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
        cigarettes: List[DetectionItem] = [
            det
            for det in all_detections
            if (det.class_id in self.phone_class_ids and det.tracker_id is not None)
        ]

        if not (heads and persons and cigarettes):
            self.ttc_counter += 1
        else:
            head = largest_volume(heads)
            person = find_person_from_head(head, persons)
            if self.detect_criteria(cigarettes, head, person):
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
        cigarettes: List[DetectionItem],
        head: HeadPoseDetectionItem,
        person: HumanPoseDetectionItem,
    ) -> bool:
        """Detects the criteria of event and save event to event_window.
        Do not directly emit event on this function (we need to store on window-basis)

        Args:
            cigarettes (List[DetectionItem]): Candidate objects for match and identify event
            head (HeadPoseDetectionItem): Target head to identify event
            person (HumanPoseDetectionItem): Target person to identify event

        Returns:
            bool: True if event occurs, or False. Mandatory for clearing event window.
        """
        distances = [head.abs_distance(phone) for phone in cigarettes]
        if not distances or min(distances) > self.distance_thresh:
            return False

        matched_phone = cigarettes[argmin(distances)]
        self.event_window.append(
            SmokeEvent(
                ptype=self.ptype,
                det_phone=matched_phone,
                det_head=head,
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
