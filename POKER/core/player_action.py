from dataclasses import dataclass
from enum import Enum, auto


class ActionType(Enum):
    """
    Enumeration of all legal player actions during a betting round.
    """
    FOLD = auto()
    CHECK = auto()
    CALL = auto()
    RAISE = auto()


@dataclass(frozen=True)
class PlayerAction:
    """
    Immutable value object representing a player's intent.
    This is created by the UI (or AI) and consumed by the engine.
    """
    action_type: ActionType.CHECK
    player_index: int = 0
    raise_amount: int = 0
