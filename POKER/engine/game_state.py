# engine/game_state.py
from enum import Enum, auto

class GamePhase(Enum):
    SETUP = auto()
    PREFLOP = auto()
    FLOP = auto()
    TURN = auto()
    RIVER = auto()
    SHOWDOWN = auto()
    GAMEOVER = auto()