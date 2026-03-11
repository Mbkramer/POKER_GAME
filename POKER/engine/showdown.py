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
            
            # Handle phevaluator hand values
            if isinstance(self.best_hand_value, int):
                hand_rank = self.best_hand_value
            else: # Handle tuple hand values
                hand_rank = self.best_hand_value[0]
                self.best_hand_name = HAND_RANK_NAMES[hand_rank].upper()
        
    def find_winning_hands(self):
        
        # Muck check - if only 1 player eligible for a pot, 
        # they win by default and we can skip hand evaluation for that pot
        for pot in self.table.pots:
            if len(self.table.pots[0]["eligible"]) == 1:
                self.table.players[self.table.pots[0]["eligible"][0].id].muck = True
            elif len(self.table.pots[0]["eligible"]) > 1:
                break

        # Global best across all visible showdown hands, only for UI/debug display
        global_best_value = None
        global_best_combo = []
        global_winners = []

        self.winners_pots = []

        pot_live_money = self.table.live_money // len(self.table.pots)

        for pot in self.table.pots:

            live_money_share = pot_live_money // len(pot["eligible"])
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

                # Assign hand to player 
                # Assign hand value and cards to player if it's their current hand best 
                if not player.hand_value or hand_value > player.hand_value:
                    player.assign_hand(hand_value, sorted(best_five_card_combo))
                    # Assign hand value and cards to player if it's their current game best
                    if player.best_hand_value is None or hand_value > player.best_hand_value:
                       player.assign_best_hand(hand_value, sorted(best_five_card_combo))

                if pot_best_value is None or hand_value > pot_best_value:
                    pot_best_value = hand_value
                    pot_best_combo = sorted(best_five_card_combo)
                    pot_winners = [player]
                elif hand_value == pot_best_value:
                    pot_winners.append(player)

                # Track overall best purely for display/debugging
                if global_best_value is None or hand_value > global_best_value:
                    global_best_value = hand_value
                    global_best_combo = sorted(best_five_card_combo)
                    global_winners = [player]
                elif hand_value == global_best_value:
                    if player not in global_winners:
                        global_winners.append(player)

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
        
        