from dataclasses import dataclass
from typing import List, Dict

from core.player import Player
from core.player_action import PlayerAction, ActionType
from core.table_state import TableState

class BettingRound:
    """
    Controls a single betting round (preflop, flop, turn, river).

    Responsibilities:
    - Enforce betting order
    - Apply player actions
    - Track current bet
    - Detect round completion
    """

    def __init__(self, table: TableState, starting_index: int):
        self.table = table
        self.current_index = starting_index

        while self.table.players[self.current_index].folded or self.table.players[self.current_index].all_in:
            self.current_index += 1
            if self.current_index == self.table.num_players:
                self.current_index = 0

        self.last_raiser_index: int | None = None
        self.active = True

    def apply(self, action: PlayerAction) -> None:

        if not self.active:
            raise RuntimeError("Betting round already complete")

        if action.player_index != self.current_index:
            raise RuntimeError("Action out of turn")

        player = self.table.players[action.player_index]
        to_call = self.table.current_bet - player.bet

        match action.action_type:
            case ActionType.FOLD:
                player.fold()

            case ActionType.CHECK:
                if to_call != 0:
                    raise RuntimeError("Cannot check when facing a bet")

            case ActionType.CALL:
                self._call(player, to_call)

            case ActionType.RAISE:
                self._raise(player, to_call, action.raise_amount)
                self.last_raiser_index = action.player_index

            case _:
                raise ValueError("Invalid action type")

        self.table.players[action.player_index].touched = True
        self._check_complete()
        self._advance_turn()

    def _call(self, player: Player, to_call: int) -> None:
        if to_call <= 0:
            return
        posted = player.place_bet(to_call)
        self.table.pot += posted

    def _raise(self, player: Player, to_call: int, raise_amount: int) -> None:
        if raise_amount <= 0:
            raise ValueError("Raise amount must be positive")
        if raise_amount <= self.table.current_bet:
            raise ValueError("Raise amount must be double current bet")

        total = to_call + raise_amount
        posted = player.place_bet(total)

        self.table.current_bet = player.bet
        self.table.pot += posted

    def _advance_turn(self) -> None:
        """
        Advances to the next player who is able to act.
        Skips folded and all-in players.
        """
        count = len(self.table.players)

        for _ in range(count):
            self.current_index = (self.current_index + 1) % count
            p = self.table.players[self.current_index]
            if not p.folded and not p.all_in:
                return

    def _check_complete(self) -> None:
        """
        Betting round is complete when:
        - Only one active (non-folded) player remains, OR
        - All active players have matched the current bet
        """
        active = []

        for player in self.table.players:
            if (not player.folded) and (not player.all_in):
                if (player.bet < self.table.current_bet) or (not player.touched):
                    active.append(player.id)

        if len(active) < 1:
            self.active = False
            return

        self.active = True


# =========================
# Side pot resolution
# =========================

def resolve_side_pots(players: List[Player]) -> List[Dict]:
    """
    Breaks total contributions into main pot + side pots.

    Returns a list of pots in order:
    [
        { "amount": int, "eligible": List[Player] },
        ...
    ]
    """
    contributions = {
        p: p.current_bet
        for p in players
        if p.current_bet > 0
    }

    pots: List[Dict] = []

    while contributions:
        min_bet = min(contributions.values())
        eligible = list(contributions.keys())

        pot_amount = min_bet * len(eligible)
        pots.append({
            "amount": pot_amount,
            "eligible": eligible.copy()
        })

        for p in list(contributions.keys()):
            contributions[p] -= min_bet
            if contributions[p] == 0:
                del contributions[p]

    return pots
