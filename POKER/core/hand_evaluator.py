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

    def evaluate_5_card_hand(self, cards: list[Card]):

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
                HAND_RANKS["TRIPLES"],
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
    
    # Much faster
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
            best_hand_value, sim = self.evaluate_7_card_hand(full_hand)
            hand_rank = best_hand_value[0]
            
            #Increment the appropriate counter
            for rank_name, rank_value in HAND_RANKS.items():
                if hand_rank == rank_value:
                    hand_outs[rank_name] += 1
                    break
        try:
            hand_probabilities = self.calculate_hand_probabilities(hand_outs, num_unknown_cards)
        except Exception as e:
            print("Calculate Hand Probs Error occured")
            print(e)

        return hand_probabilities
            
    def calculate_hand_probabilities(self, hand_outs, num_unknown_cards):

        hand_probabilities = {
            "HIGH": ((hand_outs["HIGH"])/num_unknown_cards),
            "PAIR": ((hand_outs["PAIR"])/num_unknown_cards),
            "TWO_PAIR": ((hand_outs["TWO_PAIR"])/num_unknown_cards),
            "TRIPLES": ((hand_outs["TRIPLES"])/num_unknown_cards),
            "STRAIGHT": ((hand_outs["STRAIGHT"])/num_unknown_cards),
            "FLUSH": ((hand_outs["FLUSH"])/num_unknown_cards),
            "FULL_HOUSE": ((hand_outs["FULL_HOUSE"])/num_unknown_cards),
            "QUADS": ((hand_outs["QUADS"])/num_unknown_cards),
            "STRAIGHT_FLUSH": ((hand_outs["STRAIGHT_FLUSH"])/num_unknown_cards)
            }
        
        return hand_probabilities