from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np

from base.types import DetectionResult
from ..core.detections import DetectionItem

from .base import Processor


class SBPProcessor(Processor):
    def __init__(
        self,
        padded_resize: bool = False,
        image_size: Tuple[int, int] = (256, 192),
        conf_thresh: float = 0.25,
        num_keypoints: int = 11,  # 11 for PIS and 17 for COCO
    ):
        self.padded_resize = padded_resize
        if padded_resize:
            raise NotImplementedError("SBPProcessor: PaddedResize not implemented")
        self.target_size = image_size
        self.conf_thresh = conf_thresh
        self.num_keypoints = num_keypoints

    def get_input_size(self) -> Tuple[int, int]:
        return self.target_size

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        # 1. Resize
        # 2. Standardize
        # 3. ToTensor
        # 4. Normalize

        # 1. Resize
        target_height, target_width = self.target_size
        image = cv2.resize(image, (target_width, target_height))

        # 2. Standardize
        image = image.astype(np.float32) / 255.0

        # 3. ToTensor (cvtcolor, transpose, expand_dims)
        tensor = image[..., ::-1].transpose(2, 0, 1)[np.newaxis, :]

        # 4. Normalize (CHW order, RGB order)
        # std = np.array([0.229, 0.224, 0.225])
        # mean = np.array([0.485, 0.456, 0.406])
        #  tensor -= mean.reshape((-1, 1, 1))
        # tensor /= std.reshape((-1, 1, 1))

        return tensor

    def postprocess(
        self,
        outputs: np.ndarray,
        human_det: DetectionItem,
    ) -> Union[List[List[float]], Any]:
        heatmaps = outputs[0][0]  # [num_poses, 64, 48]
        _, hmap_height, hmap_width = heatmaps.shape

        # array with all of -1 values
        joints = np.zeros((self.num_keypoints, 3)) - 1

        xmin, ymin, xmax, ymax = human_det.bbox()
        box_width = xmax - xmin
        box_height = ymax - ymin

        for pose_index, pose_heatmaps in enumerate(heatmaps):
            yy, xx = np.where(pose_heatmaps >= self.conf_thresh)
            if len(yy) == 0:
                continue

            confidences = pose_heatmaps[yy, xx]
            argmax_idx = np.argmax(confidences)

            x_idx = xx[argmax_idx]
            y_idx = yy[argmax_idx]
            conf = confidences[argmax_idx]

            pos_x = x_idx / hmap_width * box_width + xmin
            pos_y = y_idx / hmap_height * box_height + ymin

            joints[pose_index] = np.array([pos_x, pos_y, conf])
        return joints
