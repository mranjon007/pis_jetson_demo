import copy
from typing import Any, Dict, List, Tuple

from engine.core.detections.base import PassengerType

from .base import BaseEvent
from .beltoff_event import BeltOffEvent, BeltOffEventProcessor
from .distraction_event import DistractionEvent, DistractionEventProcessor
from .drink_event import DrinkEvent, DrinkEventProcessor
from .drowsiness_event import DrowsinessEvent, DrowsinessEventProcessor
from .faint_event import FaintEvent, FaintEventProcessor
from .negative_wheelgrab_event import (
    NegativeWheelGrabEvent,
    NegativeWheelGrabEventProcessor,
)
from .seat_existence_event import (
    SeatExistenceEvent,
    SeatExistenceEventProcessor,
)
from .phone_event import PhoneAnswerEvent, PhoneAnswerEventProcessor
from .smoke_event import SmokeEvent, SmokeEventProcessor
from loguru import logger


class EventSerializer:
    """Serialize event for each frame (self.current) and also for single video file (self.cumulates)"""

    def __init__(self, event_types: List[str]):
        self.current = {event_name: False for event_name in event_types}
        self.cumulates = copy.deepcopy(self.current)  # Shallow copy

    def update_current(self, event: BaseEvent):
        if isinstance(event, BeltOffEvent):
            self.cumulates["beltoff"] = self.current["beltoff"] = True
        if isinstance(event, DistractionEvent):
            self.cumulates["distraction"] = self.current["distraction"] = True
        if isinstance(event, DrinkEvent):
            self.cumulates["drink"] = self.current["drink"] = True
        if isinstance(event, DrowsinessEvent):
            self.cumulates["drowsiness"] = self.current["drowsiness"] = True
        if isinstance(event, FaintEvent):
            self.cumulates["faint"] = self.current["faint"] = True
        if isinstance(event, NegativeWheelGrabEvent):
            self.cumulates["negative_wheelgrab"] = self.current[
                "negative_wheelgrab"
            ] = True
        if isinstance(event, PhoneAnswerEvent):
            self.cumulates["phone_answer"] = self.current["phone_answer"] = True
        if isinstance(event, SmokeEvent):
            self.cumulates["smoke"] = self.current["smoke"] = True
        if isinstance(event, SeatExistenceEvent):
            self.cumulates["seat_existence"] = self.current["seat_existence"] = True

    def serialize_current(self):
        return copy.deepcopy(self.current)

    def serialize_cumulates(self):
        return copy.deepcopy(self.cumulates)

    def clear_current(self):
        for event_name in self.current.keys():
            self.current[event_name] = False


EVENT_PARAM_PROCESSORS = {
    "phone_answer": PhoneAnswerEventProcessor,
    "smoke": SmokeEventProcessor,
    "beltoff": BeltOffEventProcessor,
    "distraction": DistractionEventProcessor,
    "drink": DrinkEventProcessor,
    "drowsiness": DrowsinessEventProcessor,
    "faint": FaintEventProcessor,
    "negative_wheelgrab": NegativeWheelGrabEventProcessor,
    "seat_existence": SeatExistenceEventProcessor,
}

EVENT_PARAM_DIM_INCLUDE_PROCESSORS = ["faint", "negative_wheelgrab"]


def create_event_processors(
    event_params: Dict[str, Any],
    src_dims: Tuple[int, int],
) -> Tuple[Dict[str, BaseEvent], Dict[str, List[str]]]:
    event_processors = {}
    events_per_passengers = {
        passenger_type: list(event_details.keys())
        for (passenger_type, event_details) in event_params.items()
    }

    for passenger_type, event_details in event_params.items():
        for event_type, event_opts in event_details.items():
            if passenger_type == "driver":
                target_ptype = PassengerType.DRIVER
            elif passenger_type == "passenger":
                target_ptype = PassengerType.PASSENGER
            else:
                target_ptype = PassengerType.NOT_SET

            if event_type in EVENT_PARAM_DIM_INCLUDE_PROCESSORS:
                extra_event_opts = {"src_dims": src_dims}
                extra_event_opts.update(event_opts)
                event_opts = extra_event_opts

            if event_type in EVENT_PARAM_PROCESSORS:
                event_name = f"{passenger_type}_{event_type}"  # driver_phone
                event_processors[event_name] = EVENT_PARAM_PROCESSORS[event_type](
                    ptype=target_ptype, **event_opts
                )
            else:
                logger.warning(
                    "Event processor of event {event_type} is not registered."
                )

    return event_processors, events_per_passengers
