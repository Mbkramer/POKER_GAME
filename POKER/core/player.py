from dataclasses import dataclass
from typing import List
from core.card import Card
from core.card import HAND_RANKS

#########
# PLAYER CLASS
#########
class Player:

    def __init__(self, id, cash, buy_in):
        self.id = id
        self.order = id
        self.cash = cash
        self.bet = 0
        self.hand = []
        self.buy_in = buy_in

        # Game play variables
        self.folded = False
        self.all_in = False
        self.playing = True
        self.touched = False

        # End game stats
        self.game_hand_value = None
        self.game_best_hand: List[Card]

        self.hand_probabilities = {
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
        
        self.best_hand_probs = []

        #end game stats
        self.best_hand_value = 0
        self.best_hand = []
        self.largest_potshare = 0
        self.cash_by_round = [cash]

    def deal(self, Card):
        self.hand.append(Card)

    def place_bet(self, bet):
        self.bet += bet
        self.cash -= bet
        return bet
    
    def fold(self):
        self.folded = True
        self.bet = 0
    
    def rake(self, pot_share):
        self.cash += pot_share

    def show_hand(self):
        hand = self.hand
        
        hand_string = "\n"

        for Card in hand:
            hand_string = hand_string + Card.value + " of " + Card.suit + "\n"
        
        return hand_string
    
    def assign_hand_value(self, value):
        self.hand_value = value
    
    def clear_hand(self):
        self.bet = 0
        self.all_in = False
        self.folded = False

    def finished(self):
        self.playing = False
        self.bet=0
        self.cash=0

    def best_hands_probability(self):

        hand_probs = []
        increment = .60

        for i in range (4):

            increment = increment - (.15 * i)
            
            if self.hand_probabilities["STRAIGHT_FLUSH"] > .15:
                hand_prob = {"HAND": "STRAIGHT_FLUSH", "PROBABILITY": self.hand_probabilities["STRAIGHT_FLUSH"]}
                hand_probs.append(hand_prob)
            if self.hand_probabilities["QUADS"] * HAND_RANKS["QUADS"] > increment and self.hand_probabilities["QUADS"] > .05:
                hand_prob = {"HAND": "QUADS", "PROBABILITY": self.hand_probabilities["QUADS"]}
                hand_probs.append(hand_prob)
            if self.hand_probabilities["FULL_HOUSE"] * HAND_RANKS["FULL_HOUSE"] > increment and self.hand_probabilities["FULL_HOUSE"] > .05:
                hand_prob = {"HAND": "FULL_HOUSE", "PROBABILITY": self.hand_probabilities["FULL_HOUSE"]}
                hand_probs.append(hand_prob)
            if self.hand_probabilities["FLUSH"] * HAND_RANKS["FLUSH"] > increment and self.hand_probabilities["FLUSH"] > .05:
                hand_prob = {"HAND": "FLUSH", "PROBABILITY": self.hand_probabilities["FLUSH"]}
                hand_probs.append(hand_prob)
            if self.hand_probabilities["STRAIGHT"] * HAND_RANKS["STRAIGHT"] > increment and self.hand_probabilities["STRAIGHT"] > .05:
                hand_prob = {"HAND": "STRAIGHT", "PROBABILITY": self.hand_probabilities["STRAIGHT"]}
                hand_probs.append(hand_prob)
            if self.hand_probabilities["TRIPLES"] * HAND_RANKS["TRIPLES"] > increment and self.hand_probabilities["TRIPLES"] > .05:
                hand_prob = {"HAND": "TRIPLES", "PROBABILITY": self.hand_probabilities["TRIPLES"]}
                hand_probs.append(hand_prob)
            if self.hand_probabilities["TWO_PAIR"] * HAND_RANKS["TWO_PAIR"] > increment and self.hand_probabilities["TWO_PAIR"] > .05:
                hand_prob = {"HAND": "TWO_PAIR", "PROBABILITY": self.hand_probabilities["TWO_PAIR"]}
                hand_probs.append(hand_prob)
            if i <= 1:
                if self.hand_probabilities["PAIR"] * HAND_RANKS["PAIR"] > increment and self.hand_probabilities["PAIR"] > .05:
                    hand_prob = {"HAND": "PAIR", "PROBABILITY": self.hand_probabilities["PAIR"]}
                    hand_probs.append(hand_prob)

        # Remove duplicates, keeping only the first occurrence of each hand type 
        seen_items = {}
        distinct_hand_probs = []

        for hand_prob in hand_probs:
            if hand_prob["HAND"] not in seen_items:
                distinct_hand_probs.append(hand_prob)
                seen_items[hand_prob["HAND"]] = hand_prob

                distinct_hand_probs = list(seen_items.values())

        self.best_hand_probs = distinct_hand_probs.copy()

    def print_hand_probabilities(self):
        print(f"PLAYER {self.id} HAND PROBABILITIES\n")
        print(f"HIGH: {self.hand_probabilities["HIGH"]}")
        print(f"PAIR: {self.hand_probabilities["PAIR"]}")
        print(f"TWO_PAIR: {self.hand_probabilities["TWO_PAIR"]}")
        print(f"TRIPLES: {self.hand_probabilities["TRIPLES"]}")
        print(f"STRAIGHT: {self.hand_probabilities["STRAIGHT"]}")
        print(f"FLUSH: {self.hand_probabilities["FLUSH"]}")
        print(f"FULL_HOUSE: {self.hand_probabilities["FULL_HOUSE"]}")
        print(f"QUADS: {self.hand_probabilities["QUADS"]}")
        print(f"STRAIGHT_FLUSH: {self.hand_probabilities["STRAIGHT_FLUSH"]}\n")
    
    def __repr__(self) -> str:
        return (
            f"Player(name={self.id}, "
            f"wallet={self.cash}, "
            f"bet={self.bet}, "
            f"folded={self.folded}, "
            f"all_in={self.all_in})"
        )
    