from typing import List, Tuple, Union

import numpy as np

from .detections import DetectionItem, HeadPoseDetectionItem, HumanPoseDetectionItem
from .detections.humanpose import PassengerType

HumanHeadDetectionItem = Union[HumanPoseDetectionItem, HeadPoseDetectionItem]


def divide_passenger(
    detection_items: List[DetectionItem],
    input_shape: Tuple[int, int],
    accept_threshold: float = 0.1,
    backseat_size_threshold: Tuple[int, int] = [110, 110],
    is_reversed: bool = False,
) -> Tuple[List[HumanHeadDetectionItem], List[HumanHeadDetectionItem],]:
    """Divide passengers by [Passenger, Driver] group.
    divided passenger group will be saved into `passenger_type` attribute of
    `HumanPoseDetectionItem` object.

    Args:
        detection_items (List[DetectionItem]): All detection items
        input_shape (Tuple[int, int]): Source image shape in (height, width) order
        accept_threshold (str): Accept threshold for center split.
        reversed (bool): Divide in reverse order (Driver sits in left side).

    Returns:
        Tuple[List[DetectionItem], List[DetectionItem]]: Divided group in (drivers, passengers) order.
    """
    for det in detection_items:
        divide_passenger_single_(
            det,
            input_shape=input_shape,
            accept_threshold=accept_threshold,
            backseat_face_threshold=backseat_size_threshold,
            is_reversed=is_reversed,
        )

    drivers: List[HumanHeadDetectionItem] = [
        item
        for item in detection_items
        if isinstance(item, (HumanPoseDetectionItem, HeadPoseDetectionItem))
        and item.passenger_type == PassengerType.DRIVER
    ]
    passengers: List[HumanHeadDetectionItem] = [
        item
        for item in detection_items
        if isinstance(item, (HumanPoseDetectionItem, HeadPoseDetectionItem))
        and item.passenger_type == PassengerType.PASSENGER
    ]

    return drivers, passengers


def divide_passenger_single_(
    det: HumanHeadDetectionItem,
    input_shape: Tuple[int, int],
    accept_threshold: float = 0.1,
    backseat_face_threshold: Tuple[int, int] = [140, 140],
    backseat_person_threshold: Tuple[int, int] = [500, 550],
    is_reversed: bool = False,
) -> None:
    img_height, img_width = input_shape

    if not isinstance(det, (HumanPoseDetectionItem, HeadPoseDetectionItem)):
        return

    # Divide front passenger
    if np.all(det.bbox()[::2] < img_width * (0.5 + accept_threshold)):
        if is_reversed:
            det.passenger_type = PassengerType.DRIVER
        else:
            det.passenger_type = PassengerType.PASSENGER
    elif np.all(det.bbox()[::2] > img_width * (0.5 - accept_threshold)):
        if is_reversed:
            det.passenger_type = PassengerType.PASSENGER
        else:
            det.passenger_type = PassengerType.DRIVER

    # Divide backseat passenger headpose
    if isinstance(det, (HeadPoseDetectionItem)):
        min_face_height, min_face_width = backseat_face_threshold
        width, height = det.bbox()[2:] - det.bbox()[:2]
        if not (
            width * (1 + accept_threshold) >= min_face_width
            and height * (1 + accept_threshold) >= min_face_height
        ):
            det.passenger_type = PassengerType.PASSENGER_BACKSEAT

    # Divide backseat passenger humanpose
    if isinstance(det, (HumanPoseDetectionItem)):
        min_person_height, min_person_width = backseat_person_threshold
        width, height = det.bbox()[2:] - det.bbox()[:2]
        if not (
            width * (1 + accept_threshold) >= min_person_width
            and height * (1 + accept_threshold) >= min_person_height
        ):
            det.passenger_type = PassengerType.PASSENGER_BACKSEAT
