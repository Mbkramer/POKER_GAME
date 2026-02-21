from dataclasses import dataclass
from typing import List, Dict
from pathlib import Path
import random
from datetime import datetime
import json
import os

from core.hand_evaluator import HAND_RANKS, CARD_VALUE
from core.card import SUIT_VALUE
from core.deck import Deck
from core.player_action import PlayerAction, ActionType
from core.table_state import TableState
from engine.betting import BettingRound
from engine.showdown import Showdown
from engine.game_state import GamePhase
from bots.cfr_bots.neural.state_encoder import get_profile

# store directory
OUT_DIRETORY = 'data/live_play_policy_distribution'

#Bot net paths
TWO_PLAYER = 'POKER/bots/cfr_bots/checkpoints/best_2P_10B_200W.pt'
FOUR_PLAYER = 'POKER/bots/cfr_bots/checkpoints/best_4P_10B_500W.pt'
SIX_PLAYER = 'POKER/bots/cfr_bots/checkpoints/best_6P_10B_500W.pt'

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
        self.muck = False

        self.hand_counter = 0
        self.phh: Dict = {}
        self.store_path = ''
        self.init_phh_store()

    # =========================
    # Bots
    # =========================

    def init_bots(self, num_players):
        from bots.game_bots.game_neural_bot import NeuralPokerBot, get_neural_bot
        from bots.game_bots.hybrid_bot import HybridPokerBot, get_hybrid_bot
        
        bot_net_path = FOUR_PLAYER
        if num_players == 2:
            bot_net_path = TWO_PLAYER
        elif num_players <= 4:
            bot_net_path = FOUR_PLAYER
        elif num_players <= 6:
            bot_net_path = SIX_PLAYER
        else:
            print(f"Error loading {num_players} player net path...\nDefault to 4 player")
        
        player_index = random.randint(0, num_players-1)

        for player in self.table.players:
            if player.id == player_index:
                continue

            agression = 1.5 
            player.is_bot = True
            player.bot = get_hybrid_bot(bot_net_path, player.id, self.table, None, agression)

    def _update_bots(self):

        if self.phase != GamePhase.SHOWDOWN:
            for player in self.table.players:
                if player.is_bot:
                    player.bot.update(self.table, self.phase)

    def _reset_bots(self):
        for player in self.table.players:
            if player.is_bot:
                player.bot.reset_hand(self.phase)

    # =========================
    # Poker Hand Histories Store
    # =========================

    def init_phh_store(self):
        
        try:
            self.store_path = OUT_DIRETORY + f"/{self.table.num_players}P{self.table.buy_in}B{self.table.wallet}W_T:{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
            os.makedirs(self.store_path)
        except Exception as e:
            print(e)

    def write_phh(self):

        try:
            # Use 'a' mode to open the file for appending
            with open(self.store_path+f"/{self.hand_counter}.json", 'w') as json_file:
                json.dump(self.phh, json_file, indent=4)
            print(f"Hand appended to {self.store_path} successfully.") #
        except Exception as e:
            print(f"Error: {e}") #

    def load_phh(self):

        phh = {
            "variant": "NT",
            "ante_trimming_status": "false",
            "antes": [0] * self.table.num_players,
            "blinds_or_straddles": [0] * self.table.num_players, # need to add small and big blind bet sizes
            "min_bet": self.table.buy_in,
            "starting_stacks": [0] * self.table.num_players,
            "actions": [],
            "hand": self.hand_counter,
            "seats": [0] * self.table.num_players,
            "players": [0] * self.table.num_players,
            "finishing_stacks": [0] * self.table.num_players
        }

        phh['blinds_or_straddles'][self.table.small_blind] = self.table.buy_in/2
        phh['blinds_or_straddles'][self.table.big_blind] = self.table.buy_in

        for player in self.table.players:
            phh['starting_stacks'][player.id] = player.cash
            phh['seats'][player.id] = player.id
            phh['players'][player.id] = str(player.id)

        self.phh = phh


    # =========================
    # Hand setup
    # =========================

    def start_hand(self) -> None:

        self.hand_counter+=1
        self.load_phh()

        self.deck.shuffle()
        self._post_blinds()
        self._deal_hole_cards()
        self.muck = False

        self.phase = GamePhase.PREFLOP
        
        for player in self.table.players:
            get_profile(player.id).record_hand_start(player.cash)
        
        self._reset_bots()
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

        "d dh pN <card><card>"

        player_deals = [""] * self.table.num_players_playing

        for i in range(2):
            for p in self.table.players:
                p.muck = False
                if i == 0 and p.playing:
                    player_deals[p.id] = f"d dh p{p.id} "
                if len(p.hand) < 2 and p.playing:
                    card = self.deck.deal()
                    player_deals[p.id]+=str(card.id)
                    p.deal(card)
        
        for deal in player_deals:
            self.phh['actions'].append(deal)
        

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
                player.hand_bet = 0

                get_profile(player.id).record_hand_end(
                    player.cash,
                    reached_showdown=(self.phase == GamePhase.SHOWDOWN)
                )

                while len(player.hand) > 0:
                    self.deck.cards.append(player.hand.pop())

        while len(self.table.community_cards) > 0:
            self.deck.cards.append(self.table.community_cards.pop())

        while len(self.table.burn_deck) > 0:
            self.deck.cards.append(self.table.burn_deck.pop())

        if len(self.deck.cards) != 52:
            raise ValueError(f"Deck should have 52 cards. DECK LENGTH: {len(self.deck.cards)}")
        
        self.table.pot = 0
        self.table.live_money = 0
        self.table.current_bet = 0
        self.last_raise_size = 0
        self.table.n_raises = 0
        self.table.end_hand = False
        self.pots: List[Dict] = []

        # record hand
        for player in self.table.players:
            self.phh['finishing_stacks'][player.id] = player.cash
        self.write_phh()
        
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

        board_deal = f"d db "

        for _ in range(count):
            card = self.deck.deal()
            board_deal+=card.id
            self.table.community_cards.append(card)

        self.phh['actions'].append(board_deal)

    # deal a card to the table burn deck
    def _deal_burn(self) -> None:
        self.table.burn_deck.append(self.deck.deal())
        
    # process ui game action    
    def apply_action(self, action) -> None:

        if not self.betting_round or not self.betting_round.active:
            raise RuntimeError("No active betting round")
        
        # Record to profile BEFORE applying
        profile = get_profile(action.player_index)
        
        if self.phase == GamePhase.PREFLOP:
            if action.action_type in (ActionType.CALL, ActionType.RAISE):
                profile.record_vpip()
            if action.action_type == ActionType.RAISE:
                profile.record_pfr()
                # 3-bet: raise when there's already been a raise this street
                if self.table.n_raises >= 1:
                    profile.record_3bet()
                else:
                    profile.record_3bet_opp()
            # ATS: raise from steal position (BTN/CO/SB) when folded to
            if action.action_type == ActionType.RAISE:
                steal_seats = [
                    self.table.dealer_index,  # BTN
                    (self.table.dealer_index - 1) % self.table.num_players,  # CO
                    self.table.small_blind,    # SB
                ]
                if action.player_index in steal_seats:
                    profile.record_ats_opp(attempted=True)
        else:  # postflop
            aggressive = action.action_type == ActionType.RAISE
            profile.record_postflop_action(aggressive)
            
            # C-bet tracking
            if self.phase == GamePhase.FLOP:
                profile.record_cbet_faced(folded=action.action_type == ActionType.FOLD)

        # store phh action
        if action.action_type == ActionType.FOLD:
            self.phh['actions'].append(f"p{action.player_index} f")
        elif action.action_type == ActionType.CALL or ActionType.CHECK:
            self.phh['actions'].append(f"p{action.player_index} cbr {action.raise_amount}") # Check
        elif action.action_type == ActionType.RAISE:
            self.phh['actions'].append(f"p{action.player_index} cc")

        #execute game action in betting round
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
                self._run_monte_carlo_predictions()

            case GamePhase.FLOP:
                self._deal_burn()
                self._deal_community(1)
                self.phase = GamePhase.TURN
                # Apply monte_carlo_hand_probabilities
                self._run_monte_carlo_predictions()

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
                    
        self._update_bots()
        self._start_betting_round()

    def _run_monte_carlo_predictions(self):

        # Apply monte_carlo_hand_probabilities
        for player in self.table.players:
            if player.playing:
                hand_probabilities = self.evaluator.evaluate_monte_carlo_hand_probabilities(self.phase, self.table, player)
                player.hand_probabilities = hand_probabilities
                player.best_hands_probability()

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

        cash_in_game=self.table.pot
        for player in self.table.players:
            cash_in_game+=player.cash
        
        live_money_share = self.table.live_money / len(self.winning_players) #Share of folded money

        for player in self.table.players:
            for pot in self.winners_pots:
                for winner in pot['winners']:
                    if winner.id == player.id:

                        if pot['best_hand_value'][0] > player.best_hand_value:
                            player.best_hand_value = pot['best_hand_value'][0]
                            player.best_hand = pot['best_five_card_combo']

                        if pot['amount'] > player.largest_potshare:
                            player.largest_potshare = pot['amount']

                        if player.muck:
                            self.muck = True

                        player.rake(pot['amount'])

        for player in self.winning_players:
            player.rake(live_money_share)

        cash_in_game = 0
        for player in self.table.players:
            cash_in_game += player.cash

        self.reset_round()
