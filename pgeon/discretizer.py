import abc
from typing import Union, Sequence, List
from enum import Enum
from typing import TypeVar, Sequence

_Enum = TypeVar('_Enum', bound=Enum)
class Predicate:
    def __init__(self, predicate: Union[Enum, type], value: Union[Sequence[Union[Enum, int]], Enum, int]):
        self.predicate = predicate
        # Ensure value is always stored as a list for consistency
        if isinstance(value, (Enum, int)):
            self.value: List[Union[Enum, int]] = [value]
        else:
            self.value: List[Union[Enum, int]] = list(value)
            

    def __eq__(self, other):
        if not isinstance(other, Predicate):
            return False
        return self.predicate == other.predicate and self.value == other.value

    def __str__(self):
        # Handle both Enums and ints in value
        values_str = ",".join(self.format_value(val) for val in self.value)
        return f'{self.predicate.__name__}({values_str})'

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(str(self))

    def __lt__(self, other):
        if not isinstance(other, Predicate):
            raise ValueError("Cannot compare Predicate with non-Predicate type.")
        else:
            return hash(self.predicate) < hash(other.predicate)

    @staticmethod
    def format_value(val):
        if isinstance(val, Enum):
            return val.name  # For Enum members, use the name
        else:
            return str(val)  # For integers or other types, convert directly to string

class Discretizer(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(subclass, 'discretize') \
           and callable(subclass.discretize) \
           and hasattr(subclass, 'state_to_str') \
           and callable(subclass.state_to_str) \
           and hasattr(subclass, 'str_to_state') \
           and callable(subclass.str_to_state) \
           and hasattr(subclass, 'nearest_state') \
           and callable(subclass.nearest_state)

    @abc.abstractmethod
    def discretize(self, state):
        pass

    @abc.abstractmethod
    def state_to_str(self, state) -> str:
        pass

    @abc.abstractmethod
    def str_to_state(self, state: str):
        pass

    @abc.abstractmethod
    def nearest_state(self, state):
        pass

    @abc.abstractmethod
    def all_actions(self):
        pass

    @abc.abstractmethod
    def get_predicate_space(self):
        pass
