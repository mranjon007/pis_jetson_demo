from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union

import numpy as np

from engine import TRTInfer


class Processor(ABC):
    __instance__ = None

    @classmethod
    def get_instance(cls):
        if not cls.__instance__:
            cls.__instance__ = cls()
        return cls.__instance__

    @abstractmethod
    def get_input_size(self) -> Tuple[int, int]:
        """Get model input image size.

        Returns:
            Tuple[int, int]: image size in [width, height] format.
        """

    def warmup_engine(self, engine: TRTInfer, duration: int = 1) -> None:
        """Warmup engine for `duration` batches.

        Args:
            engine (TRTInfer): Engine to warm up (Engine must match processor's input size)
        """
        input_size = self.get_input_size()
        assert (
            len(input_size) == 2
        ), "Input size impl must be in [width, height] format."
        assert isinstance(input_size[0], int) and isinstance(
            input_size[1], int
        ), "Input size impl must follow [int, int] type format."

        image = (np.random.randn(*input_size, 3) * 255).astype(np.uint8)
        tensor = self.preprocess(image)
        for _ in range(duration):
            _ = engine.infer(tensor)

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def postprocess(self, outputs: np.ndarray) -> Union[List[List[float]], Any]:
        pass
