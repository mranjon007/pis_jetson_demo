from typing import Tuple
import numpy as np


class HeadPoseCalibrator:
    def __init__(self):
        self.yaw_data = []
        self.pitch_data = []

    def init_calib(self, outputs: np.ndarray) -> None:
        pitch, yaw, roll = outputs

        self.yaw_data.append(yaw)
        self.pitch_data.append(pitch)

    def _calc_yaw_pitch_mean(self):
        yaw_mean = 0 if not self.yaw_data else np.mean(self.yaw_data)
        pitch_mean = 0 if not self.pitch_data else np.mean(self.pitch_data)
        return yaw_mean, pitch_mean

    def calib(self, outputs: np.ndarray) -> Tuple[float, float]:
        pitch, yaw, roll = outputs
        yaw_mean, pitch_mean = self._calc_yaw_pitch_mean()

        yaw_calib = abs(yaw - yaw_mean)
        pitch_calib = abs(pitch - pitch_mean)

        return yaw_calib, pitch_calib
