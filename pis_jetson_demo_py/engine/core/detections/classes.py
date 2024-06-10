import re
from typing import List


class ClassNamesManager:
    __instance__ = None

    @classmethod
    def get_instance(cls):
        if not cls.__instance__:
            raise RuntimeError(
                "You must initialize ClassesManager before getting instance"
            )
        return cls.__instance__

    @classmethod
    def create_instance(cls, **kwargs):
        cls.__instance__ = cls(**kwargs)
        return cls.__instance__

    def __init__(self, class_names: List[str]):
        self.classes = {
            class_id: class_name.lower()
            for (class_id, class_name) in enumerate(class_names)
        }
        assert len(set(self.classes.values())) == len(
            list(self.classes.values())
        ), "Duplicate class name exists"

    def name(self, class_id: int) -> str:
        assert (
            class_id in self.classes.keys()
        ), f"Class ID {class_id} not found on classes (We have {len(self.classes)} classes)."
        return self.classes[class_id]

    def id(self, class_name: str) -> int:
        class_name = class_name.lower()
        assert (
            class_name in self.classes.values()
        ), f"Class name {class_name} not found on classes."
        for key, value in self.classes.values():
            if value == class_name:
                return key
        raise RuntimeError("Inconsistency error")

    def find_class_ids(self, target_regex: str) -> List[int]:
        class_ids = [
            class_id
            for (class_id, class_name) in self.classes.items()
            if re.findall(target_regex, class_name)
        ]

        if len(class_ids) == 0:
            raise RuntimeError(f"Class name {target_regex} not found on classes.")
        return class_ids

    def get_face_class_ids(self) -> List[int]:
        return self.find_class_ids("face")

    def get_person_class_ids(self) -> List[int]:
        return self.find_class_ids("(person|men)")

    def get_phone_class_ids(self) -> List[int]:
        return self.find_class_ids("(phone|cell)")

    def get_smoke_class_ids(self) -> List[int]:
        return self.find_class_ids("(smok|ciga|cigg)")

    def get_belt_class_ids(self) -> List[int]:
        return self.find_class_ids("(belt)")

    def get_drink_class_ids(self) -> List[int]:
        return self.find_class_ids(
            "(bottle|drink|water|soda|coke|cola|pepsi|juice|beer|wine|alcohol)"
        )

    def get_open_eye_class_ids(self) -> List[int]:
        return self.find_class_ids("eye_open")

    def get_closed_eye_class_ids(self) -> List[int]:
        return self.find_class_ids("eye_closed")
