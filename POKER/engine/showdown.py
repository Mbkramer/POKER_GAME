from typing import List, Dict
from core.card import Card
from core.card import CARD_VALUE
from core.hand_evaluator import  HAND_RANKS, HAND_RANK_NAMES
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
        self.best_hand_name = None
        self.best_five_card_combo = []

        self.find_winning_hands()

        hand_rank = "HIGH"

        if self.best_hand_value is not None:
            if isinstance(self.best_hand_value, int):
                hand_rank = self.best_hand_value
            else:
                hand_rank = self.best_hand_value[0]
                self.best_hand_name = HAND_RANK_NAMES[hand_rank].upper()
        
    def find_winning_hands(self):
        
        for pot in self.table.pots:
            if len(self.table.pots[0]["eligible"]) == 1:
                self.table.players[self.table.pots[0]["eligible"][0].id].muck = True
            elif len(self.table.pots[0]["eligible"]) > 1:
                break

        global_best_value = None
        global_best_combo = []
        global_winners = []

        self.winners_pots = []

        # Distribute live_money (folded chips) into the first eligible pot only.
        remaining_live_money = self.table.live_money

        for pot in self.table.pots:

            n_eligible = len(pot["eligible"])
            if remaining_live_money > 0 and n_eligible > 0:
                live_money_share = remaining_live_money // n_eligible
                remaining_live_money = 0
            else:
                live_money_share = 0

            # amount is always defined for every pot iteration
            amount = pot["amount"] + live_money_share

            eligible_players = [p for p in pot["eligible"] if p.playing and not p.folded]

            if not eligible_players:
                continue

            pot_best_value = None
            pot_best_combo = []
            pot_winners = []

            for player in eligible_players:
                seven_card_hand = list(self.table.community_cards) + list(player.hand)
                hand_value, best_five_card_combo = self.hand_evaluator.evaluate_7_card_hand(seven_card_hand)

                if not player.hand_value or hand_value > player.hand_value:
                    player.assign_hand(hand_value, sorted(best_five_card_combo))
                    if player.best_hand_value is None or hand_value > player.best_hand_value:
                       player.assign_best_hand(hand_value, sorted(best_five_card_combo))

                if pot_best_value is None or hand_value > pot_best_value:
                    pot_best_value = hand_value
                    pot_best_combo = sorted(best_five_card_combo)
                    pot_winners = [player]
                elif hand_value == pot_best_value:
                    pot_winners.append(player)

                if global_best_value is None or hand_value > global_best_value:
                    global_best_value = hand_value
                    global_best_combo = sorted(best_five_card_combo)
                    global_winners = [player]
                elif hand_value == global_best_value:
                    if player not in global_winners:
                        global_winners.append(player)

            # Guard: pot_winners should always be populated after the loop above,
            # but skip if somehow empty to avoid ZeroDivisionError.
            if not pot_winners:
                continue

            pot_share = amount // len(pot_winners)
            remainder = amount % len(pot_winners)

            self.winners_pots.append({
                "amount": pot_share,
                "remainder": remainder,
                "winners": pot_winners.copy(),
                "best_hand_value": pot_best_value,
                "best_five_card_combo": pot_best_combo
            })

        self.best_hand_value = global_best_value
        self.best_five_card_combo = global_best_combo
        self.winning_players = global_winners