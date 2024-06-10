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


class DrowsinessEvent(BaseEvent):
    def __init__(
        self,
        ptype: PassengerType,
        det_closed_eye: DetectionItem,
        det_head: HeadPoseDetectionItem,
        det_person: HumanPoseDetectionItem,
    ):
        self.ptype: PassengerType = ptype
        self.det_closed_eye: DetectionItem = det_closed_eye
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


class DrowsinessEventProcessor(BaseEventCallbackProcessor):
    LOG_HEADER: str = "[Drowsiness] "

    def __init__(
        self,
        ttc: int,
        tte: int,
        ptype: PassengerType,
    ):
        """Processes drowsiness event.
        Calculate intersection between eye and person's head
        to recognize if the specific person is drowsing (closed eyes) or
        concentrating on driving (open eyes).

        Args:
            ttc (int): Time To Clear (event unseen counts until clearing existing events)
            tte (int): Time To Emit (event seen counts for making alarm/event/report)
            ptype (PassengerType): Specific passenger type to be watched.
        """
        super(DrowsinessEventProcessor, self).__init__()

        self.event_window = []
        self.ttc = ttc
        self.tte = tte
        self.ptype = ptype
        self.ttc_counter = 0

        self.closed_eye_class_ids = (
            ClassNamesManager.get_instance().get_closed_eye_class_ids()
        )

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
        closed_eyes: List[DetectionItem] = [
            det
            for det in all_detections
            if (
                det.class_id in self.closed_eye_class_ids and det.tracker_id is not None
            )
        ]

        if not (heads and persons and closed_eyes):
            self.ttc_counter += 1
        else:
            head = largest_volume(heads)
            person = find_person_from_head(head, persons)
            if self.detect_criteria(closed_eyes, head, person):
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
        closed_eyes: List[DetectionItem],
        head: HeadPoseDetectionItem,
        person: HumanPoseDetectionItem,
    ) -> bool:
        """Detects the criteria of event and save event to event_window.
        Do not directly emit event on this function (we need to store on window-basis)

        Args:
            closed_eyes (List[DetectionItem]): Candidate objects for match and identify event
            head (HeadPoseDetectionItem): Target head to identify event
            person (HumanPoseDetectionItem): Target person to identify event

        Returns:
            bool: True if event occurs, or False. Mandatory for clearing event window.
        """
        if person is None:
            return False

        intersections = [person.intersection(closed_eye) for closed_eye in closed_eyes]
        closed_eye = closed_eyes[argmax(intersections)]
        if not intersections or max(intersections) == 0:
            self.event_window.append(
                DrowsinessEvent(
                    ptype=self.ptype,
                    det_closed_eye=closed_eye,
                    det_head=head,
                    det_person=person,
                )
            )
            return True

        # We may need to identify the eye currently person has
        # but event will not be stored inside HumanPoseDetectionItem.
        return False

    def send_event(self) -> None:
        # Check the thresholds and counts of the events
        # whether this event message should be emitted.

        if len(self.event_window) >= self.tte:
            if not self.emit(self.event_window[-1]):
                logger.warning("Failed to emit event to one of handlers.")
            else:
                logger.debug("Successfully sent event")
