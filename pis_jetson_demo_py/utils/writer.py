# Resembles cv2.VideoWriter
import numpy as np


class DummyWriter:
    """Dummy cv2.VideoWriter for None Output"""

    def __init__(self, url=None):
        self._opened = True

    def write(self, frame: np.ndarray) -> bool:
        return True

    def isOpened(self) -> bool:
        return self._opened

    def release(self) -> bool:
        self._opened = False
        return True
