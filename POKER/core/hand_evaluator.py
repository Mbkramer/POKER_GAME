from collections import Counter
from itertools import combinations
import random

from functools import lru_cache
from core.card import CARD_VALUE, HAND_RANKS, HAND_RANK_NAMES, VALUE_NAME
from core.card import Card
from core.deck import Deck
from core.table_state import TableState
from engine.game_state import GamePhase

import pandas as pd
import json 
from pathlib import Path

class HandEvaluator:

    def __init__(self):
        pass

    @lru_cache(maxsize=16384) # 1MB
    def _evaluate_5_card_hand_cached(self, card_tuple):

        values = []
        suits = []

        for value, suit in card_tuple:
            values.append(value)
            suits.append(suit)

        values.sort(reverse=True)
        value_counter = Counter(values)

        # Flush check
        is_flush = False
        for suit in suits:
            if suit != suits[0]:
                is_flush = False
                break
            elif suit == suits[0]:
                is_flush = True

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

        # Straight Flush
        if is_straight and is_flush:
            return (HAND_RANKS["STRAIGHT_FLUSH"], straight_high)
        
        # Check Grouped Values 
        grouped_values = sorted(
            value_counter.items(),
            key=lambda x: (-x[1], -x[0])
        )

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
            return (HAND_RANKS["FLUSH"], tuple(values))  # Convert to tuple for hashability

        # Straight
        if is_straight:
            return (HAND_RANKS["STRAIGHT"], straight_high)

        # Three of a Kind
        if grouped_values[0][1] == 3:
            kickers = []
            for i in range(1, len(grouped_values)):
                kickers.append(grouped_values[i][0])

            return (
                HAND_RANKS["TRIPLES"],
                grouped_values[0][0],
                tuple(kickers)  # Convert to tuple for hashability
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
                tuple(kickers)  # Convert to tuple for hashability
            )

        # High Card
        return (HAND_RANKS["HIGH"], tuple(values))  # Convert to tuple for hashability

    def evaluate_5_card_hand(self, cards: list[Card]):

        # Convert cards to a sorted tuple of (value, suit) pairs for consistent caching
        card_tuple = tuple(sorted(
            ((CARD_VALUE[card.value], card.suit) for card in cards),
            reverse=True
        ))
        
        result = self._evaluate_5_card_hand_cached(card_tuple)
        
        # Convert tuples back to lists for backward compatibility
        if isinstance(result[-1], tuple) and len(result[-1]) > 1 and isinstance(result[-1][0], int):
            # This is a list of kickers or values, convert back to list
            return result[:-1] + (list(result[-1]),)
        return result
    
    def evaluate_7_card_hand(self, cards):

        best_hand_value = None
        best_5_card_combo = []

        for five_card_combo in combinations(cards, 5):
            current_value = self.evaluate_5_card_hand(five_card_combo)

            if best_hand_value is None or current_value > best_hand_value:
                best_hand_value = current_value
                best_5_card_combo = five_card_combo

        return best_hand_value, best_5_card_combo
    
    # Calculate player hand probabailities at flop and turn
    def evaluate_monte_carlo_hand_probabilities(self, phase, table, player):

        num_unknown_cards = 52 - 2 - len(table.community_cards)
        num_simulations = 5000

        if phase == GamePhase.FLOP:
            cards_to_deal = 2
        elif phase == GamePhase.TURN:
            cards_to_deal = 1
        elif phase == GamePhase.RIVER:
            cards_to_deal = 0
        else:
            raise ValueError(f"Invalid phase: {phase}")

        known_cards = []
        for card in player.hand:
            known_cards.append(Card(card.id, card.suit, card.value))
        for card in table.community_cards:
            known_cards.append(Card(card.id, card.suit, card.value))

        unknown_cards = Deck()

        for card in known_cards:
            unknown_cards.cards.remove(card)

        hand_outs = {
            "HIGH": 0,
            "PAIR": 0,
            "TWO_PAIR": 0,
            "TRIPLES": 0,
            "STRAIGHT": 0,
            "FLUSH": 0,
            "FULL_HOUSE": 0,
            "QUADS": 0,
            "STRAIGHT_FLUSH": 0
        }

        for sim in range(num_simulations):
            # Randomly sample future cards
            sampled_cards = random.sample(unknown_cards.cards, cards_to_deal)
            full_hand = known_cards + sampled_cards
            
            # Evaluate best 5-card hand
            best_hand_value, _ = self.evaluate_7_card_hand(full_hand)
            hand_rank = best_hand_value[0]
            
            #Increment the appropriate counter
            for rank_name, rank_value in HAND_RANKS.items():
                if hand_rank == rank_value:
                    hand_outs[rank_name] += 1
                    break
        try:
            hand_probabilities = self.calculate_hand_probabilities(hand_outs, num_simulations) # from num unknown cards..
        except Exception as e:
            print("Calculate Hand Probs Error occured")
            print(e)

        return hand_probabilities
            
    def calculate_hand_probabilities(self, hand_outs, num_simulations):

            raw = {k: v / num_simulations for k, v in hand_outs.items()}

            # Cumulative: "at least X" probabilities
            hand_probabilities = {
                "HIGH":          1.0,
                "PAIR":          raw["PAIR"] + raw["TWO_PAIR"] + raw["TRIPLES"] + raw["STRAIGHT"] + raw["FLUSH"] + raw["FULL_HOUSE"] + raw["QUADS"] + raw["STRAIGHT_FLUSH"],
                "TWO_PAIR":      raw["TWO_PAIR"] + raw["TRIPLES"] + raw["STRAIGHT"] + raw["FLUSH"] + raw["FULL_HOUSE"] + raw["QUADS"] + raw["STRAIGHT_FLUSH"],
                "TRIPLES":       raw["TRIPLES"] + raw["FULL_HOUSE"] + raw["QUADS"] + raw["STRAIGHT_FLUSH"],
                "STRAIGHT":      raw["STRAIGHT"] + raw["STRAIGHT_FLUSH"],
                "FLUSH":         raw["FLUSH"] + raw["STRAIGHT_FLUSH"],
                "FULL_HOUSE":    raw["FULL_HOUSE"] + raw["QUADS"] + raw["STRAIGHT_FLUSH"],
                "QUADS":         raw["QUADS"] + raw["STRAIGHT_FLUSH"],
                "STRAIGHT_FLUSH": raw["STRAIGHT_FLUSH"],
            }

            return hand_probabilities