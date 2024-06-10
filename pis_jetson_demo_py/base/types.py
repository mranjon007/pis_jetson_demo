from enum import Enum
from typing import Tuple

# Has 6 detection results: [xmin, ymin, xmax, ymax, conf, class_id]
DetectionResult = Tuple[int, int, int, int, float, int]

# [cumulative_detections, iou_between_rep_and_current]
TrackInfo = Tuple[int, float]


class BlobCropExpandType(Enum):
    NONE = 0
    EXACT_PIXEL = 1
    PERCENTAGE = 2


EVENT_TYPES = [
    "phone_answer",
    "smoke",
    "distraction",
    "beltoff",
    "drink",
    "drowsiness",
    "faint",
    "negative_wheelgrab",
    "seat_existence",
]
