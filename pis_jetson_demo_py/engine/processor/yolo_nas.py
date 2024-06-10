from typing import List, Tuple, Union

import cv2
import numpy as np

from utils.nms import non_max_suppression

from .base import Processor


class YoloNASProcessor(Processor):
    def __init__(
        self,
        padded_resize: bool = True,
        standardize: bool = False,
        target_size: Tuple[int, int] = (640, 640),
        conf_thresh: float = 0.55,
        nms_iou_thresh: float = 0.6,
        soft_nms_iou_thresh: float = 0.2,
    ):
        self.padded_resize: bool = padded_resize
        self.ratio: Tuple[float, float] = 1.0, 1.0
        # [Left, Top, Right, Bottom]
        self.pads: Union[List[int], None] = None
        self.pad_value: int = 114

        self.standardize: bool = standardize
        self.target_size: Tuple[int, int] = target_size
        self.conf_thresh: float = conf_thresh
        self.nms_thresh: float = nms_iou_thresh
        self.soft_nms_thresh: float = (
            soft_nms_iou_thresh  # Separated class NMS threshold
        )
        self.class_separated_nms: bool = False

    def get_input_size(self) -> Tuple[int, int]:
        return self.target_size

    def calculate_size(self, raw_height: int, raw_width: int) -> Tuple[int, int]:
        """Calculate size from raw_width and raw_height (original image size)
        to rescaled width and height which fits on self.target_size frame

        Args:
            raw_height (int): Original image's height
            raw_width (int): Original image's width

        Returns:
            Tuple[int, int, Tuple[float, float]]: New height, width rescaled with ratio, and also ratio.
        """
        target_height, target_width = self.target_size
        ratio = min(target_height / raw_height, target_width / raw_width)
        new_height = int(raw_height * ratio)
        new_width = int(raw_width * ratio)
        return new_height, new_width, (ratio, ratio)

    def pad_image(
        self,
        image: np.ndarray,
        input_shape: Tuple[int, int],
        output_shape: Tuple[int, int],
    ) -> np.ndarray:
        # Get center padding coordinates
        # see: super_gradients.training.transforms.utils._get_center_padding_coordinates
        pad_height, pad_width = (
            output_shape[0] - input_shape[0],
            output_shape[1] - input_shape[1],
        )

        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top

        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        self.pads = [pad_left, pad_top, pad_right, pad_bottom]

        # Pad image
        # see: super_gradients.training.transforms.utils._pad_image
        pad_h = (pad_top, pad_bottom)
        pad_w = (pad_left, pad_right)

        if len(image.shape) == 3:
            return np.pad(
                image,
                (pad_h, pad_w, (0, 0)),
                "constant",
                constant_values=self.pad_value,
            )
        else:
            return np.pad(
                image, (pad_h, pad_w), "constant", constant_values=self.pad_value
            )

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        # 1. Pad-aware Resize
        # 2. Pad
        # 3. Standardize - disabled
        # 4. ToTensor

        # 1. Pad-aware Resize
        height, width, _ = image.shape

        # 2. Pad
        if self.padded_resize:
            new_height, new_width, self.ratio = self.calculate_size(height, width)
            input_shape: Tuple[int] = (new_height, new_width)
            output_shape: Tuple[int] = self.target_size
            image = cv2.resize(image, (new_width, new_height))
            image = self.pad_image(image, input_shape, output_shape)
        else:
            self.ratio = (self.target_size[0] / height, self.target_size[1] / width)
            image = cv2.resize(image, self.target_size)

        # 3. Standardize - Disabling by default due to invalid result
        if self.standardize:
            image = image.astype(np.float32) / 255.0

        # 4. ToTensor (cvtcolor, transpose, expand_dims)
        tensor = image[..., ::-1].transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)

        return tensor

    def pad_aware_shift_bboxes_(self, bboxes_xyxy: np.ndarray) -> np.ndarray:
        # see super_gradients.training.transforms.utils._shift_bboxes
        assert self.pads, "You're trying to post-process before pre-process."
        shift_h = -self.pads[1]
        shift_w = -self.pads[0]

        bboxes_xyxy[:, :, [0, 2]] += shift_w
        bboxes_xyxy[:, :, [1, 3]] += shift_h

    def rescale_bboxes_(self, bboxes_xyxy: np.ndarray) -> np.ndarray:
        # see super_gradients.training.transforms.utils._rescale_bboxes
        scale_factors = (1 / self.ratio[0], 1 / self.ratio[1])
        sy, sx = scale_factors

        bboxes_xyxy *= np.array([[[sx, sy, sx, sy]]], dtype=bboxes_xyxy.dtype)

    def postprocess(self, outputs: List[np.ndarray]) -> List[np.ndarray]:
        pred_bboxes, pred_scores, pred_class_id = outputs

        if self.padded_resize:
            # Unpad (realign)
            self.pad_aware_shift_bboxes_(pred_bboxes)

        # Resize (enlarge)
        self.rescale_bboxes_(pred_bboxes)

        # 3. NMS and filtering
        # Do NMS over all bboxes
        indicies = pred_scores[..., 0] > self.conf_thresh
        pred_bboxes = pred_bboxes[indicies]
        pred_scores = pred_scores[indicies]
        pred_class_id = pred_class_id[indicies]

        # Filter out invalid boxes containing -50 or lower value
        indicies = (pred_bboxes > -50).all(axis=1, keepdims=False)
        pred_bboxes = pred_bboxes[indicies]
        pred_scores = pred_scores[indicies]
        pred_class_id = pred_class_id[indicies]

        # xmin, ymin, xmax, ymax, conf, class_id
        pre_nms_predictions = np.concatenate(
            [pred_bboxes, pred_scores, pred_class_id], axis=1
        )

        if self.class_separated_nms:
            pred_class_id = pred_class_id[:, 0]  # Preserve last dim
            # 1. Person (class 0~5), eye (class 7~8)
            person_class_mask = np.where(pred_class_id <= 5)
            eye_class_mask = np.where(
                np.logical_and(7 <= pred_class_id, pred_class_id <= 8)
            )
            others_class_mask = np.where(
                np.logical_or(pred_class_id == 6, 9 <= pred_class_id)
            )

            person_pre_nms_predictions = pre_nms_predictions[person_class_mask]
            eye_pre_nms_predictions = pre_nms_predictions[eye_class_mask]
            others_pre_nms_predictions = pre_nms_predictions[others_class_mask]

            person_nms_keep_mask = non_max_suppression(
                person_pre_nms_predictions,
                iou_threshold=self.soft_nms_thresh,
                class_agnostic=True,
            )
            eye_nms_keep_mask = non_max_suppression(
                eye_pre_nms_predictions,
                iou_threshold=self.soft_nms_thresh,
                class_agnostic=True,
            )
            person_post_nms_predictions = person_pre_nms_predictions[
                person_nms_keep_mask
            ]
            eye_post_nms_predictions = eye_pre_nms_predictions[eye_nms_keep_mask]

            # 2. Merge with others and do NMS again
            pre_nms_predictions = np.concatenate(
                [
                    person_post_nms_predictions,
                    eye_post_nms_predictions,
                    others_pre_nms_predictions,
                ],
                axis=0,
            )

            nms_keep_mask = non_max_suppression(
                pre_nms_predictions, iou_threshold=self.nms_thresh
            )
            post_nms_predictions = pre_nms_predictions[nms_keep_mask]

            pred_bboxes = post_nms_predictions[..., :4]
            pred_scores = post_nms_predictions[..., 4:5]
            pred_class_id = post_nms_predictions[..., 5:6].astype(int)

        else:
            nms_keep_mask = non_max_suppression(
                pre_nms_predictions, iou_threshold=self.nms_thresh
            )

            post_nms_predictions = pre_nms_predictions[nms_keep_mask]

            pred_bboxes = post_nms_predictions[..., :4]
            pred_scores = post_nms_predictions[..., 4:5]
            pred_class_id = post_nms_predictions[..., 5:6].astype(int)

        return pred_bboxes, pred_scores, pred_class_id
