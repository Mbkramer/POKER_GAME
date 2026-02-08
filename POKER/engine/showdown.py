from typing import List, Dict
from core.card import Card
from core.card import CARD_VALUE, HAND_RANKS, HAND_RANK_NAMES
from core.player import Player
from core.table_state import TableState
from core.hand_evaluator import HandEvaluator

from collections import Counter
from itertools import combinations

class Showdown():

    def __init__(self, table: TableState, hand_evaluator: HandEvaluator):

        self.table = table

        self.pots_winning_players = []
        self.winners_pots: List[Dict] = []

        self.hand_evaluator = hand_evaluator
        self.winning_players = []
        self.best_hand_value = None
        self.best_five_card_combo = []
        self.pot_share = 0

        self.find_winning_hands()

        hand_rank = "HIGH"

        if self.best_hand_value:
            hand_rank = self.best_hand_value[0]
        self.best_hand_name = HAND_RANK_NAMES[hand_rank].upper()

    def find_winning_hands(self):

        for pot in self.table.pots:

            amount = pot['amount']
            eligible_players = pot['eligible']

            for player in eligible_players:
                if player.folded == True:
                    continue

                seven_card_hand = []

                for card in self.table.community_cards:
                    seven_card_hand.append(card)
                for card in player.hand:
                    seven_card_hand.append(card)

                hand_value, best_five_card_combo = self.hand_evaluator.evaluate_7_card_hand(seven_card_hand)
                player.assign_hand_value(hand_value)

                if self.best_hand_value is None or hand_value[0] > self.best_hand_value[0]:

                    self.best_hand_value = hand_value
                    self.best_five_card_combo = sorted(best_five_card_combo)
                    self.winning_players = [player]
                    self.pots_winning_players = [player]

                elif hand_value == self.best_hand_value:
                    if player not in self.winning_players:
                        self.winning_players.append(player)
                        self.pots_winning_players.append(player)

            pot_share = amount // len(self.pots_winning_players)

            self.winners_pots.append({
                "amount": pot_share,
                "winners": self.pots_winning_players.copy(),
                "best_hand_value": self.best_hand_value,
                "best_five_card_combo": self.best_five_card_combo
            })
    
    