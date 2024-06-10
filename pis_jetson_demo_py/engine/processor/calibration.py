from typing import Any, List, Tuple, Union

import numpy as np

from .base import Processor


class HeadPoseCalibrationProcessor(Processor):
    def get_input_size(self) -> Tuple[int, int]:
        pass

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        pass

    def postprocess(self, outputs: np.ndarray) -> Union[List[List[float]], Any]:
        pass

    def init_calib(self, outputs):
        yaw_data = []
        pitch_data = []

        yaw = outputs[1]
        pitch = outputs[0]

        yaw_data.append(yaw)
        pitch_data.append(pitch)

        return yaw_data, pitch_data

    def calc_yaw_pitch_mean(self, yaw_data: np.ndarray, pitch_data: np.ndarray):
        yaw_mean = 0
        pitch_mean = 0

        yaw_mean = 0 if not yaw_data else np.mean(yaw_data)
        pitch_mean = 0 if not pitch_data else np.mean(pitch_data)

        return yaw_mean, pitch_mean

    def calib(self, outputs, yaw_mean, pitch_mean):
        yaw_calib = 0
        pitch_calib = 0

        yaw = outputs[1]
        pitch = outputs[0]

        yaw_calib = abs(yaw - yaw_mean)
        pitch_calib = abs(pitch - pitch_mean)

        return yaw_calib, pitch_calib
