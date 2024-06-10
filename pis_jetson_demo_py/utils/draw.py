from __future__ import annotations

from math import cos, sin
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from loguru import logger

from base.types import EVENT_TYPES
from engine.core.detections import (
    DetectionItem,
    HeadPoseDetectionItem,
    HumanPoseDetectionItem,
)
from engine.core.detections.classes import ClassNamesManager
from engine.core.detections.humanpose import PassengerType
from tracker.base import BaseTracker, TrackItem

COLORS = [
    (0, 0, 255),  # 0: Red
    (0, 255, 0),  # 1: Green
    (255, 0, 0),  # 2: Blue
    (255, 0, 255),  # 3: Pink
    (255, 255, 0),  # 4: Cyan
    (0, 255, 255),  # 5: Yellow
    (255, 255, 255),  # 6: White
    (0, 0, 255),  # 7: Red
    (0, 255, 0),  # 8: Green
    (255, 0, 0),  # 9: Blue
    (255, 0, 255),  # 10: Pink
    (255, 255, 0),  # 11: Cyan
    (0, 255, 255),  # 12: Yellow
    (255, 255, 255),  # 13: White
    (0, 0, 255),  # 14: Red
]

COLOR_TRACKED = (93, 222, 110)  # Lime
COLOR_UNTRACKED = (99, 103, 224)  # Pastel Red


class DrawOptions(dict):
    DEFAULT_OPTIONS_SET = dict(
        latency_info=False,
        event_stats=True,
        untracked_items=False,
        headpose_item_values=True,
        headpose_passenger=True,
        box_track_info=True,
        backseat=True,
        bboxes=True,
        headpose=True,
        humanpose=True,
        legends=True,
    )

    def __init__(self, **kwargs):
        super(DrawOptions, self).__init__(**kwargs)
        self.__dict__ = self

    def override(self, override_options: str) -> DrawOptions:
        """Override option from override options string.

        Args:
            override_options (str): String contains override options
            (formatted in key1=value1,key2=value2, ...)

        Returns:
            DrawOptions: Current options set
        """
        if not override_options:
            return self

        for option_set in override_options.lower().strip().split(","):
            if not option_set:
                continue

            key, value = option_set.strip().split("=")
            key, value = key.strip(), value.strip()
            assert (
                key in self
            ), f'Key "{key}" not supported! supported key lists: {list(self)}'
            assert value in [
                "true",
                "false",
            ], f'Value {value} of key "{key}" not supported!'

            value = value == "true"  # str to bool
            logger.debug(f"Override options set: {key}={value}")
            self[key] = value

        return self

    def strict_update(self, target: Dict[str, Any]) -> DrawOptions:
        """Strictly override option from dictionary.
        Resembles update() behaviour of dict() but strictly check
        existing key before updating

        Args:
            target (Dict[str, Any]): Dictionary containers override options

        Returns:
            DrawOptions: Current options set
        """
        for key, value in target.items():
            assert (
                key in self
            ), f'Key "{key}" not supported! supported key lists: {list(self)}'
            assert isinstance(value, bool), "boolean value is only supported!"

            logger.debug(f"Override options set: {key}={value}")
            self[key] = value

        return self


def draw_head_pose_cube(
    frame: np.ndarray,
    tdy: int,
    tdx: int,
    pitch: float,
    yaw: float,
    roll: float,
    face_box_size: int = 150,
) -> None:
    # see: sixdrepnet.utils.plot_pose_cube

    p = pitch * np.pi / 180
    y = -(yaw * np.pi / 180)
    r = roll * np.pi / 180

    if tdx != None and tdy != None:
        face_x = tdx - 0.50 * face_box_size
        face_y = tdy - 0.50 * face_box_size
    else:
        height, width = frame.shape[:2]
        face_x = width / 2 - 0.5 * face_box_size
        face_y = height / 2 - 0.5 * face_box_size

    x1 = face_box_size * (cos(y) * cos(r)) + face_x
    y1 = face_box_size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_y
    x2 = face_box_size * (-cos(y) * sin(r)) + face_x
    y2 = face_box_size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_y
    x3 = face_box_size * (sin(y)) + face_x
    y3 = face_box_size * (-cos(y) * sin(p)) + face_y

    # Draw base in red
    cv2.line(frame, (int(face_x), int(face_y)), (int(x1), int(y1)), (0, 0, 255), 3)
    cv2.line(frame, (int(face_x), int(face_y)), (int(x2), int(y2)), (0, 0, 255), 3)
    cv2.line(
        frame,
        (int(x2), int(y2)),
        (int(x2 + x1 - face_x), int(y2 + y1 - face_y)),
        (0, 0, 255),
        3,
    )
    cv2.line(
        frame,
        (int(x1), int(y1)),
        (int(x1 + x2 - face_x), int(y1 + y2 - face_y)),
        (0, 0, 255),
        3,
    )
    # Draw pillars in blue
    cv2.line(frame, (int(face_x), int(face_y)), (int(x3), int(y3)), (255, 0, 0), 2)
    cv2.line(
        frame,
        (int(x1), int(y1)),
        (int(x1 + x3 - face_x), int(y1 + y3 - face_y)),
        (255, 0, 0),
        2,
    )
    cv2.line(
        frame,
        (int(x2), int(y2)),
        (int(x2 + x3 - face_x), int(y2 + y3 - face_y)),
        (255, 0, 0),
        2,
    )
    cv2.line(
        frame,
        (int(x2 + x1 - face_x), int(y2 + y1 - face_y)),
        (int(x3 + x1 + x2 - 2 * face_x), int(y3 + y2 + y1 - 2 * face_y)),
        (255, 0, 0),
        2,
    )
    # Draw top in green
    cv2.line(
        frame,
        (int(x3 + x1 - face_x), int(y3 + y1 - face_y)),
        (int(x3 + x1 + x2 - 2 * face_x), int(y3 + y2 + y1 - 2 * face_y)),
        (0, 255, 0),
        2,
    )
    cv2.line(
        frame,
        (int(x2 + x3 - face_x), int(y2 + y3 - face_y)),
        (int(x3 + x1 + x2 - 2 * face_x), int(y3 + y2 + y1 - 2 * face_y)),
        (0, 255, 0),
        2,
    )
    cv2.line(
        frame,
        (int(x3), int(y3)),
        (int(x3 + x1 - face_x), int(y3 + y1 - face_y)),
        (0, 255, 0),
        2,
    )
    cv2.line(
        frame,
        (int(x3), int(y3)),
        (int(x3 + x2 - face_x), int(y3 + y2 - face_y)),
        (0, 255, 0),
        2,
    )


def get_human_pose_colorset(num_poses: int) -> Tuple[List, List]:
    if num_poses == 11:  # PIS
        limb_colors = [
            (0, 102, 102),  # right face
            (102, 0, 102),  # left face
            (0, 204, 0),  # right arm
            (204, 0, 0),  # left arm
            (0, 102, 0),  # right leg
            (102, 0, 0),  # left leg
            (0, 0, 0),  # others
        ]

        # [joint_idx, joint_idx, limb_color_idx]
        joint_limbs = [
            [0, 1, 1],
            [0, 2, 0],
            [1, 3, 1],
            [2, 4, 0],
            [5, 7, 3],
            [6, 8, 2],
            [7, 9, 3],
            [8, 10, 2],
            [5, 6, 6],
        ]
    elif num_poses == 17:  # COCO
        limb_colors = [
            (0, 102, 102),  # right face
            (102, 0, 102),  # left face
            (0, 204, 0),  # right arm
            (204, 0, 0),  # left arm
            (0, 102, 0),  # right leg
            (102, 0, 0),  # left leg
            (0, 0, 0),  # others
        ]

        # [joint_idx, joint_idx, limb_color_idx]
        joint_limbs = [
            [0, 1, 1],
            [0, 2, 0],
            [1, 3, 1],
            [2, 4, 0],
            [5, 7, 3],
            [6, 8, 2],
            [7, 9, 3],
            [8, 10, 2],
            [11, 13, 5],
            [12, 14, 4],
            [13, 15, 5],
            [14, 16, 4],
            [5, 6, 6],
            [5, 11, 6],
            [6, 12, 6],
            [11, 12, 6],
        ]
    else:
        raise NotImplementedError(
            f"Colorset with num_poses={num_poses} not implemeneted"
        )

    return limb_colors, joint_limbs


def draw_human_pose_skeleton(
    frame: np.ndarray,
    pose_items: Dict[int, Tuple[int, int, float]],
) -> None:
    # Stolen from pis_pose_estimations/utils/sbp_pis_utils.py@get_pis_tagged_img_sbp
    limb_colors, joint_limbs = get_human_pose_colorset(num_poses=11)

    # Draw keypoints limbs
    for jidx_a, jidx_b, color_idx in joint_limbs:
        # Given that joint is not detected (below conf. thresh.)
        if jidx_a not in pose_items.keys() or jidx_b not in pose_items.keys():
            continue

        x1, y1, c1 = pose_items[jidx_a]
        x2, y2, c2 = pose_items[jidx_b]
        if c1 < 0 or c2 < 0:
            continue

        cv2.line(frame, (x1, y1), (x2, y2), limb_colors[color_idx], 4)

    # Draw keypoints joints
    for xpos, ypos, conf in pose_items.values():
        if conf < 0:
            continue

        cv2.circle(frame, (xpos, ypos), 4, (0, 0, 255), -1)


def draw_multiline_text_(
    frame: np.ndarray,
    text_body: str,
    ptr1: Tuple[int, int],
    ptr2: Tuple[int, int],
    color: Tuple[int, int, int],
    draw_bottom: bool,
):
    xmin, ymin = ptr1
    xmax, ymax = ptr2

    lines = text_body.splitlines()
    for idx, line in enumerate(lines):
        if draw_bottom:
            pos_draw = (xmin, ymax + 29 * (idx + 1))
        else:
            pos_draw = (xmin, ymin - 8 - 29 * (len(lines) - (idx + 1)))

        cv2.putText(
            frame,
            line,
            pos_draw,  # skip bottom line
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            1,
            lineType=cv2.LINE_AA,
        )


def draw_event_stats(
    frame: np.ndarray,
    event_info: Dict[str, Dict[str, bool]],
    per_passenger_event_types: Dict[str, List[str]],
):
    passenger_events = event_info["passenger"]
    driver_events = event_info["driver"]

    passenger_event_types = per_passenger_event_types["passenger"]
    driver_event_types = per_passenger_event_types["driver"]

    osd_text = []
    for event_name in passenger_event_types:
        if passenger_events[event_name]:
            osd_text.append(f"Passenger {event_name}")

    for event_name in driver_event_types:
        if driver_events[event_name]:
            osd_text.append(f"Driver {event_name}")

    if True not in sum(
        [list(passenger_events.values()), list(driver_events.values())], []
    ):
        osd_text.append("No event")

    for index, line in enumerate(osd_text):
        cv2.putText(
            frame,
            line,
            (10, 30 + 30 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )


def draw_boxes_and_tracks(
    frame: np.ndarray,
    face_classes: List[int],
    detection_items: List[DetectionItem],
    tracker_engine: BaseTracker,
    draw_options: DrawOptions,
    event_info: Dict[str, Dict[str, bool]],
    per_passenger_event_types: Dict[str, List[str]],
) -> None:
    tracker_type: str = tracker_engine.get_tracker_type()

    for det in detection_items:
        # BEGIN draw track information and Bounding boxes
        xmin, ymin, xmax, ymax, conf, class_id, tracker_id, *_ = det.serialize()
        class_name = ClassNamesManager.get_instance().name(class_id)

        desc = (
            f"{conf * 100:.01f}% {class_name}\n"
            + f"({xmin}, {ymin}) ({xmax-xmin}x{ymax-ymin})"
        )

        # Do not draw backseat passenger when flag is not set
        if isinstance(det, (HeadPoseDetectionItem, HumanPoseDetectionItem)):
            if (
                det.passenger_type == PassengerType.PASSENGER_BACKSEAT
                and not draw_options.backseat
            ):
                continue

        if det.tracker_id is None:  # If item is not tracked
            # Only draw while flag is set
            if draw_options.untracked_items:
                if draw_options.bboxes:
                    if draw_options.box_track_info:
                        # DummyTracker doesn't wants all boxes to be tracked
                        if tracker_type != "DummyTracker":
                            desc += f"\nUntracked ({tracker_type})"

                        if isinstance(det, HumanPoseDetectionItem):
                            if det.passenger_type == PassengerType.PASSENGER:
                                lines = desc.split("\n")
                                lines[0] += " Passenger"
                                desc = "\n".join(lines)

                            elif det.passenger_type == PassengerType.DRIVER:
                                lines = desc.split("\n")
                                lines[0] += " Driver"
                                desc = "\n".join(lines)

                            elif det.passenger_type == PassengerType.PASSENGER_BACKSEAT:
                                lines = desc.split("\n")
                                lines[0] += " Backseat"
                                desc = "\n".join(lines)

                        draw_multiline_text_(
                            frame=frame,
                            text_body=desc,
                            ptr1=(xmin, ymin),
                            ptr2=(xmax, ymax),
                            color=COLOR_UNTRACKED,
                            draw_bottom=class_id in face_classes,
                        )

                    cv2.rectangle(
                        frame,
                        (xmin, ymin),
                        (xmax, ymax),
                        COLOR_UNTRACKED,
                        2,
                    )

        else:  # If items is tracked
            tracker_metrics = det.tracker_metrics
            track_item: TrackItem = tracker_engine.tracks[tracker_id]

            if draw_options.bboxes:
                if draw_options.box_track_info:
                    desc += (
                        f"\nTrack {tracker_id:02d} "
                        + f"({tracker_type}={tracker_metrics:.02f} dets={track_item.cumulative_dets})"
                    )

                    if isinstance(det, HumanPoseDetectionItem):
                        if det.passenger_type == PassengerType.PASSENGER:
                            lines = desc.split("\n")
                            lines[0] += " Passenger"
                            desc = "\n".join(lines)

                        elif det.passenger_type == PassengerType.DRIVER:
                            lines = desc.split("\n")
                            lines[0] += " Driver"
                            desc = "\n".join(lines)

                        elif det.passenger_type == PassengerType.PASSENGER_BACKSEAT:
                            lines = desc.split("\n")
                            lines[0] += " Backseat"
                            desc = "\n".join(lines)

                    draw_multiline_text_(
                        frame=frame,
                        text_body=desc,
                        ptr1=(xmin, ymin),
                        ptr2=(xmax, ymax),
                        color=COLOR_TRACKED,
                        draw_bottom=class_id in face_classes,
                    )

                cv2.rectangle(
                    frame,
                    (xmin, ymin),
                    (xmax, ymax),
                    COLOR_TRACKED,
                    2,
                )
        # END draw track information and Bounding boxes

        if draw_options.headpose and isinstance(det, HeadPoseDetectionItem):
            # Draw the head pose cube
            (
                xmin,
                ymin,
                xmax,
                ymax,
            ) = det.bbox()

            (
                pitch,
                yaw,
                roll,
            ) = det.rotation_metrics()

            face_width = xmax - xmin
            face_height = ymax - ymin
            tdx = xmin + int(0.5 * face_width)
            tdy = ymin + int(0.5 * face_height)

            if not (
                det.passenger_type == PassengerType.PASSENGER
                and not draw_options.headpose_passenger
            ):
                if draw_options.headpose_item_values:
                    distraction_event = det.distract_events()
                    if distraction_event is None:
                        desc = f"Y:{yaw:.01f} P:{pitch:.01f}: R:{roll:.01f}\nNo distraction event"
                    else:
                        front_state, yaw_state, pitch_state = distraction_event
                        desc = f"Y:{yaw:.01f} P:{pitch:.01f}: R:{roll:.01f}\nF:{front_state} Y:{yaw_state} P:{pitch_state}"

                    if det.passenger_type == PassengerType.PASSENGER:
                        desc += " (Passenger)"
                    elif det.passenger_type == PassengerType.DRIVER:
                        desc += " (Driver)"
                    elif det.passenger_type == PassengerType.PASSENGER_BACKSEAT:
                        desc += " (Backseat)"

                    logger.debug(f"Headpose [{xmin}, {ymin}, {xmax}, {ymax}]: {desc}")
                    draw_multiline_text_(
                        frame=frame,
                        text_body=desc,
                        ptr1=(xmin, ymin),
                        ptr2=(xmax, ymax),
                        color=(0, 255, 255),
                        draw_bottom=False,  # Face always draws descriptions to bottom
                    )

                draw_head_pose_cube(
                    frame=frame,
                    tdx=tdx,
                    tdy=tdy,
                    pitch=pitch,
                    yaw=yaw,
                    roll=roll,
                    face_box_size=int(min(face_width, face_height) / 1.2),
                )

        elif draw_options.humanpose and isinstance(det, HumanPoseDetectionItem):
            # Draw the human pose items
            # Stolen from pis_pose_estimations/utils/sbp_pis_utils.py@get_pis_tagged_img_sbp
            joints: Dict[int, Tuple[int, int, float]] = det.get_joints()

            draw_human_pose_skeleton(
                frame=frame,
                pose_items=joints,
            )

    # Static OSD - draw whether boxes exist or not
    if draw_options.event_stats:
        draw_event_stats(
            frame=frame,
            event_info=event_info,
            per_passenger_event_types=per_passenger_event_types,
        )


def draw_info_text(frame: np.ndarray, text: str, left: int = 4):
    for idx, line in enumerate(text.splitlines()):
        cv2.putText(
            frame,
            line,
            (left, 28 + 32 * idx),  # + 32
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
        )
