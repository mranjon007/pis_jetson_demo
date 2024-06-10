import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Union

from loguru import logger


class BaseEvent(ABC):
    """Base event structure

    - This structure contains detailed information about event.
    - This corresponds to specific business logic Task.
    """


class BaseEventCallbackProcessor(ABC):
    """Base manual event processor

    - Every frame update() must be called for proper event propagation.
    - When event emits, consumers registered with register() will be called back
      along with BaseEvent which contains detailed event body.
    """

    def __init__(self):
        super(BaseEventCallbackProcessor, self).__init__()
        self._consumers = []

    def register(self, consumer_fn: Callable):
        if consumer_fn not in self._consumers:
            self._consumers.append(consumer_fn)
        return True

    def unregister(self, consumer_fn: Callable):
        if consumer_fn not in self._consumers:
            return False
        while consumer_fn in self._consumers:
            self._consumers.remove(consumer_fn)
        return True

    @abstractmethod
    def update(self, **kwargs) -> Any:
        # Code which finds event and calls emit() with custom BaseEvent
        pass

    def emit(self, body: BaseEvent) -> bool:
        if not self._consumers:
            logger.warning(
                f"No consumer assigned to this event type: {type(self).__name__}"
            )
            return True

        all_success = True
        try:
            for consume in self._consumers:
                all_success &= consume(body)
        except Exception as e:
            logger.error(f"Consumer error: {str(e)}")
            all_success = False
        return all_success


# class BaseEventManualProcessor(ABC):
#     """Base manual event processor

#     - Every frame update() must be called for proper event propagation.
#     - When event emits, update() returns BaseEvent which contains detailed event body.
#     """

#     def __init__(self):
#         super(BaseEventManualProcessor, self).__init__()

#     @abstractmethod
#     def update(self, **kwargs) -> Union[BaseEvent, None]:
#         # Code which finds event and returns custom BaseEvent or None
#         pass


class SimpleEvent(BaseEvent):
    def __init__(self):
        super().__init__()
        # No data is fed

    def get_data(self):
        return None


class SimpleEventProcessor(BaseEventCallbackProcessor):
    def __init__(self):
        super().__init__()
        self.frame_counts = []
        self.t_begin = time.time()

    def update(self, **kwargs) -> bool:
        if not kwargs:
            # Single frame without event
            return False
        return True


if __name__ == "__main__":
    event_processor = SimpleEventProcessor()
    print(event_processor.update())
