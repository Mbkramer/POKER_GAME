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
        self.best_hand_name = ""
        self.best_five_card_combo = []
        self.pot_share = 0
        

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

        self.table.small_blind = sb
        self.table.big_blind = bb

        self.table.pot += self.table.players[sb].place_bet(self.table.buy_in/2)
        self.table.pot += self.table.players[bb].place_bet(self.table.buy_in)
        self.table.players[bb].touched = True
        self.table.players[sb].touched = True

        self.table.current_bet = self.table.buy_in

    def _deal_hole_cards(self) -> None:

        for i in range(2):
            for p in self.table.players:
                if len(p.hand) < 2:
                    p.deal(self.deck.deal())

    def reset_round(self):

        self.table.pot = 0
        self.table.current_bet = 0
        
        self.table.dealer_index += 1
        if self.table.dealer_index >= self.table.num_players:
            self.table.dealer_index = 0

        #return cards to deck and recent indicators
        for player in self.table.players:

            player.folded = False
            player.all_in = False
            player.touched = False

            player.hand_value = 0
            player.bet = 0

            self.deck.cards.append(player.hand.pop())
            self.deck.cards.append(player.hand.pop())

            #increment dealer order
            player.order += 1
            if player.order == self.table.num_players:
                player.order = 0

        while len(self.table.community_cards) > 0:
            self.deck.cards.append(self.table.community_cards.pop())

        while len(self.table.burn_deck) > 0:
            self.deck.cards.append(self.table.burn_deck.pop())

        if len(self.deck.cards) != 52:
            raise ValueError(f"Deck should have 52 cards. DECK LENGTH: {len(self.deck.cards)}")

    # =========================
    # Betting flow
    # =========================

    def _start_betting_round(self) -> None:
        start_index = (self.table.dealer_index + 2) % len(self.table.players)
        self.betting_round = BettingRound(self.table, start_index)

    def _end_betting_round(self) -> None:
        self.table.current_bet = 0
        for player in self.table.players:
            player.touched = False
        

    def apply_action(self, action) -> None:
        """
        Entry point for UI / AI.
        """
        if not self.betting_round or not self.betting_round.active:
            raise RuntimeError("No active betting round")

        self.betting_round.apply(action)

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

            case GamePhase.FLOP:
                self._deal_burn()
                self._deal_community(1)
                
                self.phase = GamePhase.TURN

            case GamePhase.TURN:
                self._deal_burn()
                self._deal_community(1)
                
                self.phase = GamePhase.RIVER

            case GamePhase.RIVER:
                
                self.phase = GamePhase.SHOWDOWN
                self._showdown()
                
            case GamePhase.SHOWDOWN:
                self.start_hand()
                
        self._start_betting_round()

    def _deal_community(self, count: int) -> None:
        for _ in range(count):
            self.table.community_cards.append(self.deck.deal())

    def _deal_burn(self) -> None:
        self.table.burn_deck.append(self.deck.deal())

    # =========================
    # Showdown
    # =========================

    def _showdown(self):

        showdown = Showdown(self.table, self.evaluator)

        self.winning_players = showdown.winning_players
        self.best_five_card_combo = showdown.best_five_card_combo
        self.pot_share = showdown.pot_share
        self.best_hand_name = showdown.best_hand_name

        for player in self.table.players:
            for winner in self.winning_players:
                if winner.id == player.id:
                    player.rake(showdown.pot_share)
        
        self.reset_round()
