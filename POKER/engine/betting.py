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

        while self.table.players[self.current_index].folded or self.table.players[self.current_index].all_in or not self.table.players[self.current_index].playing:
            self.current_index += 1
            if self.current_index == self.table.num_players:
                self.current_index = 0

        self.last_raiser_index: int | None = None
        self.active = True
        self.pots = List[Dict]


    def apply(self, action: PlayerAction) -> None:

        if not self.active:
            raise RuntimeError("Betting round already complete")

        if action.player_index != self.current_index:
            raise RuntimeError("Action out of turn")

        player = self.table.players[action.player_index]
        to_call = self.table.current_bet - player.bet

        match action.action_type:
            case ActionType.FOLD:
                self._fold(player)

            case ActionType.CHECK:
                if to_call != 0:
                    raise RuntimeError("Cannot check when facing a bet")

            case ActionType.CALL:
                self._call(player, to_call)

            case ActionType.RAISE:
                self._raise(player, action.raise_amount)
                self.last_raiser_index = action.player_index

            case _:
                raise ValueError("Invalid action type")

        self.table.players[action.player_index].touched = True
        self._check_complete()

        if not self.active:
            self._resolve_side_pots()

        if self.active:
            self._advance_turn()

    def _fold(self, player: Player) -> None:
        self.table.live_money+=player.hand_bet
        player.fold()

    def _call(self, player: Player, to_call: int) -> None:
        if to_call <= 0:
            return
        
        total = to_call

        if to_call >= player.cash:
            player.all_in = True
            total = player.cash

        posted = player.place_bet(total)
        self.table.pot += posted

    def _raise(self, player: Player, raise_to: int) -> None:

        min_raise_to = self.table.current_bet + self.table.last_raise_size 
        all_in = player.bet + player.cash  

        # Clamp to stack (all-in)
        raise_to = min(raise_to, all_in)

        is_all_in = False
        # Check legality (unless all-in)
        if raise_to == all_in:
            is_all_in = True

        if not is_all_in and raise_to < min_raise_to:
            raise ValueError("Illegal raise size")

        # Chips actually committed
        put_in = raise_to - player.bet
        posted = player.place_bet(put_in)
        self.table.pot += posted

        # Update betting state
        if raise_to > self.table.current_bet:
            self.table.last_raise_size = raise_to - self.table.current_bet
            self.table.current_bet = raise_to

    def _advance_turn(self) -> None:
        """
        Advances to the next player who is able to act.
        Skips folded and all-in players.
        """
        count = len(self.table.players)

        for index in range(count):
            self.current_index = (self.current_index + 1) % count
            p = self.table.players[self.current_index]
            if not p.folded and not p.all_in and p.playing:
                return

    def _check_complete(self) -> None:
        """
        Betting round is complete when:
        - Only one active (non-folded) player remains, OR
        - All active players have matched the current bet
        """
        active = []
        check = 0

        for player in self.table.players:

            if (not player.folded) and (not player.all_in) and (player.playing):
                if (player.bet < self.table.current_bet) or (not player.touched):
                    active.append(player.id)

            if player.folded or player.all_in or not player.playing:
                check+=1

        if len(active) < 1:
            self.active = False

            # from equal to self.table.num_players
            if check >= self.table.num_players-1:
                self.table.end_hand = True

            return 

        self.active = True

    # =========================
    # Side pot resolution
    # =========================

    def _resolve_side_pots(self) -> None:

        contributing_players = [
            p for p in self.table.players 
            if p.bet > 0 and p.playing
        ]
            
        if not contributing_players:
            return
            
        self.pots = define_side_pots(contributing_players)
        self._get_side_pots()

    def _get_side_pots(self) -> List[Dict]:
        self.table.pots = self.pots.copy()
        
# =========================
# Define side pots
# =========================

def define_side_pots(players: List[Player]) -> List[Dict]:

    contributions = {
        p: p.hand_bet  # Using new hand_bet between streets.. bet used to determine betting round eligibility
        for p in players
        if p.hand_bet > 0 
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