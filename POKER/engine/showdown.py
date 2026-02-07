from typing import List, Dict
from core.card import Card
from core.card import CARD_VALUE, HAND_RANKS, HAND_RANK_NAMES
from core.player import Player
from core.table_state import TableState
from core.hand_evaluator import HandEvaluator
from engine.betting import resolve_side_pots

from collections import Counter
from itertools import combinations

class Showdown():

    def __init__(self, table: TableState, hand_evaluator: HandEvaluator):

        self.table = table
        self.hand_evaluator = hand_evaluator
        self.winning_players = []
        self.best_hand_value = None
        self.best_five_card_combo = []
        self.pot_share = 0
        self.find_winning_hand()
        hand_rank = "HIGH"
        if self.best_hand_value:
            hand_rank = self.best_hand_value[0]
        self.best_hand_name = HAND_RANK_NAMES[hand_rank].upper()
        
    def hand_to_name(self, value):
        return HAND_RANK_NAMES.get(value, str(value))

    def find_winning_hand(self):

        for player in self.table.players:
            if player.folded == True:
                continue

            seven_card_hand = []
            for card in self.table.community_cards:
                seven_card_hand.append(card)
            for card in player.hand:
                seven_card_hand.append(card)

            hand_value, best_five_card_combo = self.evaluate_7_card_hand(seven_card_hand)
            player.assign_hand_value(hand_value)

            if self.best_hand_value is None or hand_value > self.best_hand_value:
                self.best_hand_value = hand_value
                self.best_five_card_combo = best_five_card_combo
                self.best_five_card_combo = sorted(self.best_five_card_combo)
                self.winning_players = [player]
            elif hand_value == self.best_hand_value:
                self.winning_players.append(player)

        self.pot_share = self.table.pot // len(self.winning_players)

    def evaluate_5_card_hand(self, cards):

        values = []
        suits = []

        for card in cards:
            values.append(CARD_VALUE[card.value])
            suits.append(card.suit)

        values.sort(reverse=True)
        value_counter = Counter(values)

        # Flush check
        is_flush = True
        for suit in suits:
            if suit != suits[0]:
                is_flush = False
                break

        # Straight check (Ace high or low)
        unique_values = sorted(set(values), reverse=True)
        is_straight = False
        straight_high = 0

        for i in range(len(unique_values) - 4):
            if unique_values[i] - unique_values[i + 4] == 4:
                is_straight = True
                straight_high = unique_values[i]
                break

        if not is_straight:
            if set([14, 2, 3, 4, 5]).issubset(unique_values):
                is_straight = True
                straight_high = 5

        grouped_values = sorted(
            value_counter.items(),
            key=lambda x: (-x[1], -x[0])
        )

        # Straight Flush
        if is_straight and is_flush:
            return (HAND_RANKS["STRAIGHT_FLUSH"], straight_high)

        # Four of a Kind
        if grouped_values[0][1] == 4:
            return (
                HAND_RANKS["QUADS"],
                grouped_values[0][0],
                grouped_values[1][0]
            )

        # Full House
        if grouped_values[0][1] == 3 and grouped_values[1][1] == 2:
            return (
                HAND_RANKS["FULL_HOUSE"],
                grouped_values[0][0],
                grouped_values[1][0]
            )

        # Flush
        if is_flush:
            return (HAND_RANKS["FLUSH"], values)

        # Straight
        if is_straight:
            return (HAND_RANKS["STRAIGHT"], straight_high)

        # Three of a Kind
        if grouped_values[0][1] == 3:
            kickers = []
            for i in range(1, len(grouped_values)):
                kickers.append(grouped_values[i][0])

            return (
                HAND_RANKS["THREE_PAIR"],
                grouped_values[0][0],
                kickers
            )

        # Two Pair
        if grouped_values[0][1] == 2 and grouped_values[1][1] == 2:
            high_pair = max(grouped_values[0][0], grouped_values[1][0])
            low_pair = min(grouped_values[0][0], grouped_values[1][0])
            kicker = grouped_values[2][0]

            return (
                HAND_RANKS["TWO_PAIR"],
                high_pair,
                low_pair,
                kicker
            )

        # One Pair
        if grouped_values[0][1] == 2:
            kickers = []
            for i in range(1, len(grouped_values)):
                kickers.append(grouped_values[i][0])

            return (
                HAND_RANKS["PAIR"],
                grouped_values[0][0],
                kickers
            )

        # High Card
        return (HAND_RANKS["HIGH"], values)
    
    
    def evaluate_7_card_hand(self, cards):
        best_hand_value = None
        best_5_card_combo = []

        for five_card_combo in combinations(cards, 5):
            current_value = self.evaluate_5_card_hand(five_card_combo)

            if best_hand_value is None or current_value > best_hand_value:
                best_hand_value = current_value
                best_5_card_combo = five_card_combo

        return best_hand_value, best_5_card_combo
    