from typing import Any, List, Tuple, Union

import cv2
import numpy as np

from .base import Processor


class SixDRepNetProcessor(Processor):
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size: Tuple[int, int] = target_size

    def get_input_size(self) -> Tuple[int, int]:
        return self.target_size

    def preprocess(self, image: np.ndarray) -> np.ndarray:
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
        tensor = image[..., ::-1].transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)

        # 4. Normalize (CHW order, RGB order)
        std = np.array([0.229, 0.224, 0.225])
        mean = np.array([0.485, 0.456, 0.406])
        tensor -= mean.reshape((-1, 1, 1))
        tensor /= std.reshape((-1, 1, 1))

        return tensor

    def postprocess(self, outputs: np.ndarray) -> Union[List[List[float]], Any]:
        outputs = np.clip(outputs[0][0], a_min=-90, a_max=90)  # [1, 1, 3] -> [3, ]
        return outputs  # [3] -> [pitch, yaw, roll]
