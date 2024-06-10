import time
from threading import Thread
from typing import Tuple, Union

import cv2
import numpy as np
from loguru import logger

IS_OV2311 = False


class VideoSource:
    @staticmethod
    def set_is_ov2311(is_ov2311: bool) -> None:
        global IS_OV2311
        IS_OV2311 = is_ov2311

    @staticmethod
    def get_is_ov2311() -> bool:
        global IS_OV2311
        return IS_OV2311

    @staticmethod
    def convert_ov2311(frame: np.ndarray) -> np.ndarray:
        frame = frame.reshape(1300, 1600, 2)[:, :, 1]
        frame = np.clip(
            (frame.astype(np.float32) - 16) / (255 - 16) * 255, 0, 255
        ).astype(np.uint8)
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return frame


class SequentialVideoCapture(VideoSource):
    LOG_HEADER: str = "[InputVideoStream] "

    def __init__(
        self,
        cap: cv2.VideoCapture,
        repeat_input: bool = False,
        force_resize_shape: Tuple[int, int] = None,
    ):
        self.cap = cap
        self.repeat = repeat_input

        self.running = True
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = 0
        self.force_resize_shape = force_resize_shape
        if force_resize_shape:
            self.width, self.height = force_resize_shape

        self.latencies = []

    def start(self):
        return True

    def get_frame(self, copy=True):
        t_begin = time.time()
        if not self.cap.isOpened():
            return None  # In-while stmt (timing calculations)

        ret, frame = self.cap.read()
        if self.get_is_ov2311():
            frame = self.convert_ov2311(frame)
        t_latency_ms = int((time.time() - t_begin) * 1000)

        if not ret:
            if self.repeat:
                # Rewind back to the first frame
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                logger.info(__class__.LOG_HEADER + "Rewinding video")

                ret, frame = self.cap.read()
                if self.get_is_ov2311():
                    frame = self.convert_ov2311(frame)
                self.running = ret
                return frame

            self.running = False

            logger.debug(
                __class__.LOG_HEADER + f"EOF (Total {self.total_frames} frames decoded)"
            )
            return None

        if self.force_resize_shape:
            frame = cv2.resize(frame, self.force_resize_shape)

        logger.debug(
            __class__.LOG_HEADER
            + f"VideoCapture frame fetch latency: {t_latency_ms} ms"
        )
        self.latencies.append(t_latency_ms)
        self.total_frames += 1

        if len(self.latencies) > 100:
            self.latencies = self.latencies[-100:]

        return frame

    def get_statistics(self):
        if not self.latencies:
            return 0, 0
        avg_fps = sum(self.latencies) / len(self.latencies)
        last_latency_ms = self.latencies[-1]
        return avg_fps, last_latency_ms

    def stop(self):
        return True

    def destroy(self):
        pass

    def join(self):
        return True


class ThreadedVideoCapture(VideoSource):
    LOG_HEADER: str = "[InputVideoStream] "

    def __init__(
        self,
        cap: cv2.VideoCapture,
        repeat_input: bool = False,
        fps_limit: float = -1,
        force_resize_shape: Tuple[int, int] = None,
    ):
        self.cap = cap

        self.fps: float = self.cap.get(cv2.CAP_PROP_FPS)
        self.width: int = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height: int = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps_limit: float = fps_limit

        self.running: bool = False
        self.repeat: bool = repeat_input
        self.force_resize_shape: Tuple[int, int] = force_resize_shape
        if force_resize_shape:
            self.width, self.height = force_resize_shape

        # Ensure single frame inside self.frame until thread starts
        ret, self.frame = cap.read()
        if self.get_is_ov2311():
            self.frame = self.convert_ov2311(self.frame)
        assert ret, "VideoCaptureThread: cannot read from cap"
        self.latencies = []
        self.thread = Thread(target=self.fetch_thread, daemon=True)

    def fetch_thread(self) -> None:
        if self.fps_limit != -1:
            logger.info(
                __class__.LOG_HEADER
                + f"Syncing thread (target: {self.fps_limit:.01f} fps)"
            )
        self.running = True
        total_frames = 0

        logger.info(__class__.LOG_HEADER + "Starting decoding thread")
        while self.running and self.cap.isOpened():
            t_begin = time.time()
            _ = self.cap.grab()
            ret, frame = self.cap.retrieve()
            if self.get_is_ov2311():
                frame = self.convert_ov2311(frame)
            t_latency_ms = int((time.time() - t_begin) * 1000)

            if not ret:
                if self.repeat:
                    # Rewind back to the first frame
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    logger.info(__class__.LOG_HEADER + "Rewinding video")
                    continue

                self.running = False
                break

            if self.force_resize_shape:
                frame = cv2.resize(frame, self.force_resize_shape)

            self.frame = frame  # Prevent None assignment
            self.latencies.append((t_begin, t_latency_ms))
            total_frames += 1
            logger.debug(
                __class__.LOG_HEADER
                + f"VideoCapture frame fetch latency: {t_latency_ms} ms"
            )

            if len(self.latencies) > 100:
                self.latencies = self.latencies[-100:]

            # First apply FPS limit (higher priority), then apply syncings
            if self.fps_limit != -1:
                wait_secs = max(0, (1 / self.fps_limit) - (time.time() - t_begin))
                logger.debug(f"[FPS Cap] Waiting for {int(wait_secs * 1000)} ms")
                time.sleep(wait_secs)

        logger.info(
            __class__.LOG_HEADER
            + f"Stopped decoding thread (Total {total_frames} frames decoded)"
        )

    def start(self):
        self.running = True
        return self.thread.start()

    def get_frame(self, copy=True):
        if self.frame is None:
            return None
        elif copy:
            return self.frame.copy()
        else:
            return self.frame

    def get_statistics(self):
        if not self.latencies:
            return 0, 0

        avg_latency = (self.latencies[-1][0] - self.latencies[0][0]) / len(
            self.latencies
        )
        if avg_latency == 0:
            return 0, 0

        avg_fps = 1 / avg_latency
        last_latency_ms = self.latencies[-1][1]
        return avg_fps, last_latency_ms

    def stop(self):
        self.running = False
        self.latencies.clear()
        return self.thread.join()

    def destroy(self):
        self.cap.release()

    def join(self):
        return self.thread.join()
