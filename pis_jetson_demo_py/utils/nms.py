import numpy as np


def box_iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area_a = box_area(boxes_a.T)
    area_b = box_area(boxes_b.T)

    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[:, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[:, 2:])

    area_inter = np.prod(np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)

    return area_inter / (area_a[:, None] + area_b - area_inter)


def non_max_suppression(
    predictions: np.ndarray,
    iou_threshold: float,
    class_agnostic: bool = False,
) -> np.ndarray:
    """Non-Maximum Suppression implementation in Numpy
    See: https://blog.roboflow.com/how-to-code-non-maximum-suppression-nms-in-plain-numpy/#vectorized-non-maximum-suppression

    Args:
        predictions (np.ndarray): Predicted bounding box in [N, 5] shape: [xmin, ymin, xmax, ymax, category_id]
        iou_threshold (float): Threshold to remove box under specific IoU value.

    Returns:
        np.ndarray: Keep mask for `predictions`. Can get final results with `predictions[mask]`.
    """

    rows, columns = predictions.shape

    sort_index = np.flip(predictions[:, 4].argsort())
    predictions = predictions[sort_index]

    boxes = predictions[:, :4]
    categories = predictions[:, 5]
    ious = box_iou_batch(boxes, boxes)
    ious = ious - np.eye(rows)

    keep = np.ones(rows, dtype=bool)

    for index, (iou, category) in enumerate(zip(ious, categories)):
        if not keep[index]:
            continue

        if class_agnostic:
            condition = iou > iou_threshold
        else:
            condition = (iou > iou_threshold) & (categories == category)
        keep = keep & ~condition

    return keep[sort_index.argsort()]
