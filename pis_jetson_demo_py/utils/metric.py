import math

import numpy as np


def iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """Calculate IoU between two bounding boxes.

    Args:
        bbox1 (np.ndarray): First box to compare [xmin, ymin, xmax, ymax]
        bbox2 (np.ndarray): Second box to compare [xmin, ymin, xmax, ymax]

    Returns:
        float: 0 <= IoU <= 1
    """
    if isinstance(bbox1, np.ndarray):
        bbox1 = bbox1.tolist()
    if isinstance(bbox2, np.ndarray):
        bbox2 = bbox2.tolist()

    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    max_rxmin, max_rymin = max(xmin1, xmin2), max(ymin1, ymin2)
    min_rxmax, min_rymax = min(xmax1, xmax2), min(ymax1, ymax2)

    inter_width, inter_height = max(0, min_rxmax - max_rxmin), max(
        0, min_rymax - max_rymin
    )
    inter_area = inter_width * inter_height
    union = area1 + area2 - inter_area

    return inter_area / union


def intersection(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """Calculate intersection area between two bounding boxes.

    Args:
        bbox1 (np.ndarray): First box to compare [xmin, ymin, xmax, ymax]
        bbox2 (np.ndarray): Second box to compare [xmin, ymin, xmax, ymax]

    Returns:
        float: 0 <= Intersection <= 1
    """
    if isinstance(bbox1, np.ndarray):
        bbox1 = bbox1.tolist()
    if isinstance(bbox2, np.ndarray):
        bbox2 = bbox2.tolist()

    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2

    max_rxmin, max_rymin = max(xmin1, xmin2), max(ymin1, ymin2)
    min_rxmax, min_rymax = min(xmax1, xmax2), min(ymax1, ymax2)

    inter_width, inter_height = max(0, min_rxmax - max_rxmin), max(
        0, min_rymax - max_rymin
    )
    inter_area = inter_width * inter_height
    return inter_area


def l2_distance(
    bbox1: np.ndarray, bbox2: np.ndarray, img_width: int, img_height: int
) -> float:
    """Calculate L2 distance of center point between two bounding boxes.

    Args:
        bbox1 (np.ndarray): First box to compare [xmin, ymin, xmax, ymax]
        bbox2 (np.ndarray): Second box to compare [xmin, ymin, xmax, ymax]
        img_width (int): Image width to normalize box value
        img_height (int): Image height to normalize box value

    Returns:
        float: L2 distance
    """

    if isinstance(bbox1, np.ndarray):
        bbox1 = bbox1.tolist()
    if isinstance(bbox2, np.ndarray):
        bbox2 = bbox2.tolist()

    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2

    xmin1, xmax1 = xmin1 / img_width, xmax1 / img_width
    ymin1, ymax1 = ymin1 / img_height, ymax1 / img_height
    xmin2, xmax2 = xmin2 / img_width, xmax2 / img_width
    ymin2, ymax2 = ymin2 / img_height, ymax2 / img_height

    width1, height1 = xmax1 - xmin1, ymax1 - ymin1
    width2, height2 = xmax2 - xmin2, ymax2 - ymin2

    xc1, yc1 = xmin1 + width1 // 2, ymin1 + height1 // 2
    xc2, yc2 = xmin2 + width2 // 2, ymin2 + height2 // 2

    return math.sqrt(abs(xc2 - xc1) ** 2 + abs(yc2 - yc1) ** 2)
