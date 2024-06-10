from typing import Any, List, Tuple, Union

import numpy as np

from engine.core import DetectionItem
from base.types import BlobCropExpandType, DetectionResult


class BlobCropper:
    def __init__(
        self,
        class_filter: Union[List[int], None] = None,
        crop_expand_type: BlobCropExpandType = BlobCropExpandType.NONE,
        crop_expand_amount: Any = None,
        face_head_expansion: bool = False,
    ):
        self.class_filter = class_filter
        self.expand_type = crop_expand_type
        self.expand_amount = crop_expand_amount
        self.face_head_expansion = face_head_expansion
        self._check_types(crop_expand_type, crop_expand_amount)

    def _check_types(self, expand_type: BlobCropExpandType, expand_amount: Any) -> None:
        if expand_type == BlobCropExpandType.EXACT_PIXEL:
            assert isinstance(
                expand_amount, int
            ), "EXACT_PIXEL requires value to be integer."
        if expand_type == BlobCropExpandType.PERCENTAGE:
            assert isinstance(
                expand_amount, float
            ), "PERCENTAGE requires value to be float."
            assert (
                0 <= expand_amount <= 1
            ), "PERCENTAGE requires value to belong between 0 and 1."

    def expand_bbox(self, bbox: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Expand bounding box

        Args:
            bbox (np.ndarray): Bounding box to expand its volume.
            Must be [xmin, ymin, xmax, ymax] format and can be relative/absolute.
        Returns:
            Tuple[np.ndarray, np.ndarray]: Bounding box and Amounts (in xamount, yamount, owidth, oheight order).
        """
        bbox = bbox.copy()
        width, height = (bbox[2:] - bbox[:2]).tolist()

        if self.expand_type == BlobCropExpandType.EXACT_PIXEL:
            bbox[:2] -= self.expand_amount
            bbox[2:] += self.expand_amount
            if self.face_head_expansion:
                bbox[1] -= self.expand_amount[1] * 2  # only ymin
            amount = np.array(
                [self.expand_amount, self.expand_amount, width, height]
            ).astype(bbox.dtype)
        elif self.expand_type == BlobCropExpandType.PERCENTAGE:
            xy_amounts = np.array(
                [width * self.expand_amount, height * self.expand_amount]
            ).astype(bbox.dtype)
            bbox[:2] -= xy_amounts
            bbox[2:] += xy_amounts
            if self.face_head_expansion:
                bbox[1] -= xy_amounts[1] * 2  # only ymin
            amount = np.concatenate([xy_amounts, [width, height]])
        elif self.expand_type == BlobCropExpandType.NONE:
            amount = np.array([0, 0, width, height]).astype(bbox.dtype)
        else:
            raise NotImplementedError("Metric not implemented")

        return bbox, amount

    def shrink_bbox(self, bbox: np.ndarray) -> np.ndarray:
        """Shrink bounding box

        Args:
            bbox (np.ndarray): Bounding box to shrink its volume.
            Must be [xmin, ymin, xmax, ymax] format and can be relative/absolute.
        Returns:
            np.ndarray: Bounding box.
        """
        bbox = bbox.copy()
        width, height = (bbox[2:] - bbox[:2]).tolist()

        if self.expand_type == BlobCropExpandType.EXACT_PIXEL:
            bbox[:2] += self.expand_amount
            bbox[2:] -= self.expand_amount
            amount = np.array([self.expand_amount, self.expand_amount])
        elif self.expand_type == BlobCropExpandType.PERCENTAGE:
            xy_amounts = np.array(
                [width / (1 + amount) * amount, height / (1 + amount) * amount]
            )
            bbox[:2] += xy_amounts
            bbox[2:] -= xy_amounts
            amount = xy_amounts

        return bbox, amount

    def filter_crop_blob(
        self,
        frame: np.ndarray,
        all_detections: List[DetectionItem],
    ) -> Tuple[List[Tuple[int, DetectionItem]], List[np.ndarray]]:
        filtered_dets: List[DetectionItem] = []
        cropped_blobs: List[Tuple[np.ndarray, np.ndarray]] = []

        for det_idx, det in enumerate(all_detections):
            if self.class_filter and det.class_id not in self.class_filter:
                continue
            filtered_dets.append((det_idx, det))

        image_height, image_width, _ = frame.shape
        for det_idx, det in filtered_dets:
            bbox = det.bbox()
            bbox, amount = self.expand_bbox(bbox)

            xmin, xmax = bbox[0::2].clip(0, image_width).astype(int).tolist()
            ymin, ymax = bbox[1::2].clip(0, image_height).astype(int).tolist()

            cropped_blobs.append((frame[ymin:ymax, xmin:xmax, :], amount))

            #! BEGIN DEBUG
            # Update cropped blob parameters with existing dets
            # for visualization
            # det.xmin = xmin
            # det.ymin = ymin
            # det.xmax = xmax
            # det.ymax = ymax
            #! END DEBUG

        return filtered_dets, cropped_blobs

    def shrink_pose(self, pose: np.ndarray, amount: np.ndarray) -> np.ndarray:
        """Shrink pose joints with given amounts

        Args:
            pose (np.ndarray): Pose joint lists [num_joints, 3] -> x, y, conf
            amount (np.ndarray): Amount of expanded sizes

        Returns:
            np.ndarray: _description_
        """
        pose = pose.copy()

        aw, ah, obw, obh = amount.tolist()
        width_multiplier = obw / (obw - (aw * 2))
        height_multiplier = obh / (obh - (ah * 2))

        for jidx, single_joint in enumerate(pose):
            x, y, conf = single_joint.tolist()
            if conf < 0:
                continue

            x = (x - aw) * width_multiplier
            y = (y - ah) * height_multiplier
            pose[jidx] = np.array([x, y, conf])

        return pose
