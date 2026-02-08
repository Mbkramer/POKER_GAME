from dataclasses import dataclass
from typing import List, Dict

from core.card import HAND_RANKS, SUIT_VALUE, CARD_VALUE
from core.deck import Deck
from core.table_state import TableState
from engine.betting import BettingRound
from engine.showdown import Showdown
from engine.game_state import GamePhase

class HandController:
    """
    Runs a single poker hand from blinds to showdown.
    UI or AI must feed player actions into active betting rounds.
    """

    def __init__(self, table: TableState, evaluator):
        self.table = table
        self.evaluator = evaluator
        self.deck = Deck()
        self.phase = GamePhase.SETUP
        self.betting_round: BettingRound | None = None

        self.winning_players = []
        self.final_community_cards = []
        self.best_hand_name = ""
        self.best_five_card_combo = []
        
        self.winners_pots: List[Dict]
        
    # =========================
    # Hand setup
    # =========================

    def start_hand(self) -> None:

        self.deck.shuffle()

        self._post_blinds()
        self._deal_hole_cards()

        self.phase = GamePhase.PREFLOP
        self._start_betting_round()

    # =========================
    # Blinds & dealing
    # =========================

    def _post_blinds(self) -> None:

        sb = (self.table.dealer_index) % len(self.table.players)
        bb = (self.table.dealer_index + 1)  % len(self.table.players)

        while not self.table.players[bb].playing:
            bb += 1
            if bb >= self.table.num_players:
                bb = 0

        self.table.small_blind = sb
        self.table.big_blind = bb

        self.table.pot += self.table.players[sb].place_bet(self.table.buy_in/2)
        self.table.pot += self.table.players[bb].place_bet(self.table.buy_in)
        self.table.players[bb].touched = True
        self.table.players[sb].touched = False #From true

        self.table.current_bet = self.table.buy_in

    def _deal_hole_cards(self) -> None:

        for i in range(2):
            for p in self.table.players:
                if len(p.hand) < 2 and p.playing:
                    p.deal(self.deck.deal())

    def reset_round(self):

        #return cards to deck and recent indicators
        for player in self.table.players:
            if player.playing:

                player.folded = False
                player.all_in = False
                player.touched = False
                player.cash_by_round.append(player.cash)

                if player.cash <= 0:
                    player.playing = False
                    self.table.num_players_playing-=1

                player.hand_value = 0
                player.bet = 0

                while len(player.hand) > 0:
                    self.deck.cards.append(player.hand.pop())

        while len(self.table.community_cards) > 0:
            self.deck.cards.append(self.table.community_cards.pop())

        while len(self.table.burn_deck) > 0:
            self.deck.cards.append(self.table.burn_deck.pop())

        if len(self.deck.cards) != 52:
            raise ValueError(f"Deck should have 52 cards. DECK LENGTH: {len(self.deck.cards)}")
        
        self.table.pot = 0
        self.table.current_bet = 0
        self.table.end_hand = False
        self.pots: List[Dict]
        
        self.table.dealer_index += 1
        if self.table.dealer_index >= self.table.num_players:
            self.table.dealer_index = 0

        while not self.table.players[self.table.dealer_index].playing:
            self.table.dealer_index += 1
            if self.table.dealer_index >= self.table.num_players:
                self.table.dealer_index = 0

    # =========================
    # Betting flow
    # =========================

    # establish starting index and initialize new betting round
    def _start_betting_round(self) -> None:
        start_index = (self.table.dealer_index + 2) % len(self.table.players)
        self.betting_round = BettingRound(self.table, start_index)

    # reset table betting round data
    def _end_betting_round(self) -> None:
        self.table.current_bet = 0
        for player in self.table.players:
            player.touched = False

    # deal a card to the table community deck
    def _deal_community(self, count: int) -> None:
        for _ in range(count):
            self.table.community_cards.append(self.deck.deal())

    # deal a card to the table burn deck
    def _deal_burn(self) -> None:
        self.table.burn_deck.append(self.deck.deal())
        
    # process ui game action    
    def apply_action(self, action) -> None:
        """
        Entry point for UI / AI.
        """
        if not self.betting_round or not self.betting_round.active:
            raise RuntimeError("No active betting round")

        self.betting_round.apply(action)

        if self.table.end_hand:
            self._end_betting_round()
            self._close_hand()

        if not self.betting_round.active:
            self._end_betting_round()
            self._advance_phase()

    # =========================
    # Phase transitions
    # =========================

    def _advance_phase(self) -> None:
        
        self.table.reset_bets()

        match self.phase:
            case GamePhase.PREFLOP:
                self._deal_burn()
                self._deal_community(3)
                self.phase = GamePhase.FLOP

                # Apply monte_carlo_hand_probabilities
                for player in self.table.players:
                    if player.playing:
                        hand_probabilities = self.evaluator.evaluate_monte_carlo_hand_probabilities(self.phase, self.table, player)
                        player.hand_probabilities = hand_probabilities
                        player.best_hands_probability()

                self.phase = GamePhase.FLOP

            case GamePhase.FLOP:
                self._deal_burn()
                self._deal_community(1)

                # Apply monte_carlo_hand_probabilities
                for player in self.table.players:
                    if player.playing:
                        hand_probabilities = self.evaluator.evaluate_monte_carlo_hand_probabilities(self.phase, self.table, player)
                        player.hand_probabilities = hand_probabilities
                        player.best_hands_probability()
                
                self.phase = GamePhase.TURN

            case GamePhase.TURN:
                self._deal_burn()
                self._deal_community(1)
                
                self.phase = GamePhase.RIVER

            case GamePhase.RIVER:
                
                self.phase = GamePhase.SHOWDOWN
                self._showdown()

            case GamePhase.SHOWDOWN:
                if self.table.num_players_playing > 1:
                    self.start_hand()
                elif self.table.num_players_playing == 1:
                    self.phase = GamePhase.GAMEOVER
                    self.game_over()
                
        self._start_betting_round()

    #flash hand to showdown
    def _close_hand(self):

        if self.phase == GamePhase.PREFLOP:
            self._deal_burn()
            self._deal_community(3)
            self._deal_burn()
            self._deal_community(1)
            self._deal_burn()
            self._deal_community(1)
        elif self.phase == GamePhase.FLOP:
            self._deal_burn()
            self._deal_community(1)
            self._deal_burn()
            self._deal_community(1)
        elif self.phase == GamePhase.TURN:
            self._deal_burn()
            self._deal_community(1)

        self.phase = GamePhase.RIVER

    # Last player standing has been identified
    def game_over(self):
        return

    # =========================
    # Showdown
    # =========================

    def _showdown(self):

        showdown = Showdown(self.table, self.evaluator)

        self.final_community_cards = sorted(self.table.community_cards)
        self.winning_players = showdown.winning_players

        self.best_five_card_combo = showdown.best_five_card_combo.copy()
        self.winners_pots = showdown.winners_pots.copy()
        self.best_hand_name = showdown.best_hand_name

        for player in self.table.players:
            for pot in self.winners_pots:
                for winner in pot['winners']:
                    if winner.id == player.id:

                        if pot['best_hand_value'][0] > player.best_hand_value:
                            player.best_hand_value = pot['best_hand_value'][0]
                            player.best_hand = pot['best_five_card_combo']

                        if pot['amount'] > player.largest_potshare:
                            player.largest_potshare = pot['amount']

                        player.rake(pot['amount'])
        
        self.reset_round()
