import time
import warnings
from typing import Callable

import pycuda.driver as cuda
from loguru import logger
from datetime import datetime
import os


def null_latency_logger(strategy: str, name: str, val: float) -> None:
    pass


def default_latency_logger(strategy: str, name: str, val: float) -> None:
    if val > 1000:
        logger.debug(
            f"[Latency][{strategy}][{name}] {val:.02f} ms ({val / 1000:.02f} seconds)"
        )
    else:
        logger.debug(f"[Latency][{strategy}][{name}] {val:.02f} ms")


class MeasureLatency:
    """ContextManager to measure Latency within context block.

    Details:
    - Must call measure() until the context block ends (refs will be destroyed)

    Usage:
    ```
    with MeasureLatency(name='experiment_name', strategy='cuda') as m_experiment_name:
        # Do something TensorRT-related tasks to measure CUDA latency
        lat_experiment_name = m_experiment_name.measure()
    ```
    """

    def __init__(
        self,
        name: str,
        strategy: str = "cuda",
        logger_fn: Callable[[str, str, float], None] = null_latency_logger,
        profile: bool = False,
    ):
        self.logger_fn = logger_fn
        self.name = name
        self.strategy = strategy.lower()
        self.cp_ctx = None
        if profile:
            import cProfile

            self.cp_ctx = cProfile.Profile()
        assert self.strategy in [
            "cuda",
            "pytime",
        ], "strategy muse be one of: cuda, pytime"
        self.measured = False

        if self.strategy == "cuda":
            self.begin = cuda.Event()
            self.end = cuda.Event()
        elif self.strategy == "pytime":
            self.begin = 0
            self.end = 0
        else:
            raise NotImplementedError()

    def __enter__(self):
        if self.strategy == "cuda":
            self.begin.record()
        elif self.strategy == "pytime":
            self.begin = time.time()
        else:
            raise NotImplementedError()
        if self.cp_ctx is not None:
            self.cp_ctx.enable()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if not self.measured:
            warnings.warn(
                f"[LatencyMeasure] scope '{self.name}' left unmeasured. "
                + "Check if you are not using the latency measurements."
            )
        if self.cp_ctx is not None:
            self.cp_ctx.disable()

            # Save profile results
            os.makedirs("_profile_dumps", exist_ok=True)
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.cp_ctx.dump_stats(f"_profile_dumps/p_{current_time}_{self.name}.prof")

        del self.cp_ctx
        del self.begin
        del self.end

    def measure(self) -> float:
        """Measure latency from beginning of block until now.
        Value will differ by strategy.

        Raises:
            NotImplementedError: Invalid strategy name specified.

        Returns:
            float: Measured latency in miliseconds.
        """
        if self.strategy == "cuda":
            self.end.record()
            self.end.synchronize()
            value = self.begin.time_till(self.end)
        elif self.strategy == "pytime":
            self.end = time.time()
            value = 1000 * (self.end - self.begin)
        else:
            raise NotImplementedError()

        self.measured = True
        self.logger_fn(
            self.strategy,
            self.name,
            value,
        )
        return value
