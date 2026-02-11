import random
from collections import Counter
from itertools import combinations
import sys

sys.path.append('POKER')
from core.card import Card, HAND_RANK_NAMES, HAND_RANKS, RANK_HANDS, VALUE_NAME, CARD_VALUE, SUIT_VALUE
from core.deck import Deck
from core.hand_evaluator import HandEvaluator
from core.player import Player
from core.table_state import TableState

from engine.hand_controller import HandController, GamePhase

TEST_PASS = {
    True: 'PASS',
    False: 'FAIL'
}

class tests:
    def __init__(self):
        self.set_hand_evaluator()
        self.run_hand_evaluator_tests()
        pass

    #def test_betting_flow():

    #def test_pot_resolution():

    def test_hand_prediction(self):

        no_effect = 100
        results = []

        for i in range(2, 7):
            
            print (f"\nTESTING MONTE CARLO SIMS FOR NUM PLAYERS: {i} ")
            table = TableState(i, no_effect, no_effect)
            
            hc = HandController(table, HandEvaluator())

            all_tests = []

            for j in range(10):
                print(f"\n------------ {i} PLAYERS: TEST {j} ------------")

                tests = []
                player_stats = []

                for k in range(len(hc.table.players)):
                    player_stats.append({
                    'flop_predictions': {},
                    'fop_best_hands': [], 
                    'turn_predictions': {},
                    'turn_best_hands': [], 
                    'player_hand_result': str(),
                    'result_flop_prediction': float,
                    'result_turn_prediction': float
                })

                hc.deck.shuffle()
                hc._deal_hole_cards()

                # FLOP
                hc._deal_burn()
                hc._deal_community(3)
                hc.phase = GamePhase.FLOP

                
                for k in range(len(hc.table.players)):
                    predictions = hc.evaluator.evaluate_monte_carlo_hand_probabilities(hc.phase, hc.table, hc.table.players[k])
                    hc.table.players[k].hand_probabilities = predictions
                    player_stats[k]['flop_predictions'] = predictions
                    hc.table.players[k].best_hands_probability()
                    player_stats[k]['flop_best_hands'] = hc.table.players[k].best_hand_probs

                # TURN
                hc._deal_burn()
                hc._deal_community(1)
                hc.phase = GamePhase.TURN

                for k in range(len(hc.table.players)):
                    predictions = hc.evaluator.evaluate_monte_carlo_hand_probabilities(hc.phase, hc.table, hc.table.players[k])
                    hc.table.players[k].hand_probabilities = predictions
                    player_stats[k]['turn_predictions'] = predictions
                    hc.table.players[k].best_hands_probability()
                    player_stats[k]['turn_best_hands'] = hc.table.players[k].best_hand_probs

                # RIVER
                hc._deal_burn()
                hc._deal_community(1)
                hc.phase = GamePhase.RIVER

                for k in range(len(hc.table.players)):
                    seven_cards = []

                    for card in hc.table.community_cards:
                        seven_cards.append(card)
                    for card in hc.table.players[k].hand:
                        seven_cards.append(card)

                    hand_value, best_hand = hc.evaluator.evaluate_7_card_hand(seven_cards)
                    player_stats[k]['player_hand_result'] = hand_value[0]

                i=0
                for k in range(len(hc.table.players)):
                    
                    test_pass = False

                    for best_hand in player_stats[k]['flop_best_hands']:
                        if RANK_HANDS[player_stats[k]['player_hand_result']] == best_hand['HAND']:
                            test_pass = True

                    for best_hand in player_stats[k]['turn_best_hands']:
                        if not test_pass and RANK_HANDS[player_stats[k]['player_hand_result']] == best_hand['HAND']:
                            test_pass = True
                    
                    tests.append({'test_pass': test_pass, 'hand': RANK_HANDS[player_stats[k]['player_hand_result']], 'result_flop_prediction': player_stats[k]['flop_predictions'][RANK_HANDS[player_stats[k]['player_hand_result']]], 'result_turn_prediction': player_stats[k]['turn_predictions'][RANK_HANDS[player_stats[k]['player_hand_result']]]})
                    i+=1

                for player in hc.table.players:
                    while len(player.hand)>0:
                        hc.deck.cards.append(player.hand.pop())
                while len(hc.table.community_cards)>0:
                    hc.deck.cards.append(hc.table.community_cards.pop())
                while len(hc.table.burn_deck)>0:
                    hc.deck.cards.append(hc.table.burn_deck.pop())

                all_tests.append(tests)

            results.append(all_tests)

            total_num_tests = 0
            total_num_tests_pass = 0

            total_num_tests_hands = {
                "PAIR": 0,
                "TWO_PAIR": 0,
                "TRIPLES": 0,
                "STRAIGHT": 0,
                "FLUSH": 0,
                "FULL_HOUSE": 0,
                "QUADS": 0,
                "STRAIGHT_FLUSH": 0
            }

            total_tests_predictions_flop = {
                "PAIR": [],
                "TWO_PAIR": [],
                "TRIPLES": [],
                "STRAIGHT": [],
                "FLUSH": [],
                "FULL_HOUSE": [],
                "QUADS": [],
                "STRAIGHT_FLUSH": []
            }

            total_tests_predictions_turn = {
                "PAIR": [],
                "TWO_PAIR": [],
                "TRIPLES": [],
                "STRAIGHT": [],
                "FLUSH": [],
                "FULL_HOUSE": [],
                "QUADS": [],
                "STRAIGHT_FLUSH": []
            }

            total_num_tests_pass_hands = {
                "PAIR": 0,
                "TWO_PAIR": 0,
                "TRIPLES": 0,
                "STRAIGHT": 0,
                "FLUSH": 0,
                "FULL_HOUSE": 0,
                "QUADS": 0,
                "STRAIGHT_FLUSH": 0
            }

            num_players = 2
            for all_tests in results:
                l = 0
                for tests in all_tests:
                    m=0
                    for test in tests:
                        if test['hand'] != 'HIGH':

                            if test['test_pass']:    
                                total_num_tests_pass_hands[test['hand']] += 1
                                total_tests_predictions_flop[test['hand']].append(test['result_flop_prediction'])
                                total_tests_predictions_turn[test['hand']].append(test['result_turn_prediction'])
                                total_num_tests_pass+=1

                            total_num_tests_hands[test['hand']] += 1
                            total_num_tests+=1
                        m+=1
                    l+=1
                num_players+=1

        print(f"PERCENT SUCCESSFULL TESTS = %{(total_num_tests_pass/total_num_tests)*100} ({total_num_tests_pass}/{total_num_tests})")
        print('\n')
        if total_num_tests_hands['PAIR'] != 0:
            total_flop = 0
            for prediction in total_tests_predictions_flop['PAIR']:
                total_flop+=prediction
            total_turn = 0
            for prediction in total_tests_predictions_turn['PAIR']:
                total_turn+=prediction

            print(f"PERCENT SUCCESSFULL PAIR TESTS = %{(total_num_tests_pass_hands['PAIR']/total_num_tests_hands['PAIR'])*100} ({total_num_tests_pass_hands['PAIR']}/{total_num_tests_hands['PAIR']})")
            print(f"AVERAGE PAIR FLOP PREDICTION = {(total_flop/total_num_tests_pass_hands['PAIR'])}")
            print(f"AVERAGE PAIR TURN PREDICTION = {(total_turn/total_num_tests_pass_hands['PAIR'])}")
            print('\n')

        if total_num_tests_hands['TWO_PAIR'] != 0:
            total_flop = 0
            for prediction in total_tests_predictions_flop['TWO_PAIR']:
                total_flop+=prediction
            total_turn = 0
            for prediction in total_tests_predictions_turn['TWO_PAIR']:
                total_turn+=prediction
 
            print(f"PERCENT SUCCESSFULL TWO_PAIR TESTS = %{(total_num_tests_pass_hands['TWO_PAIR']/total_num_tests_hands['TWO_PAIR'])*100} ({total_num_tests_pass_hands['TWO_PAIR']}/{total_num_tests_hands['TWO_PAIR']})")
            print(f"AVERAGE TWO_PAIR FLOP PREDICTION = {(total_flop/total_num_tests_pass_hands['TWO_PAIR'])}")
            print(f"AVERAGE TWO_PAIR TURN PREDICTION = {(total_turn/total_num_tests_pass_hands['TWO_PAIR'])}")
            print('\n')
 
        if total_num_tests_hands['TRIPLES'] != 0:
            total_flop = 0
            for prediction in total_tests_predictions_flop['TRIPLES']:
                total_flop+=prediction
            total_turn = 0
            for prediction in total_tests_predictions_turn['TRIPLES']:
                total_turn+=prediction

            print(f"PERCENT SUCCESSFULL TRIPLES TESTS = %{(total_num_tests_pass_hands['TRIPLES']/total_num_tests_hands['TRIPLES'])*100} ({total_num_tests_pass_hands['TRIPLES']}/{total_num_tests_hands['TRIPLES']})")
            print(f"AVERAGE TRIPLES FLOP PREDICTION = {(total_flop/total_num_tests_pass_hands['TRIPLES'])}")
            print(f"AVERAGE TRIPLES TURN PREDICTION = {(total_turn/total_num_tests_pass_hands['TRIPLES'])}")
            print('\n')        
        
        if total_num_tests_hands['STRAIGHT'] != 0:
            total_flop = 0
            for prediction in total_tests_predictions_flop['STRAIGHT']:
                total_flop+=prediction
            total_turn = 0
            for prediction in total_tests_predictions_turn['STRAIGHT']:
                total_turn+=prediction

            print(f"PERCENT SUCCESSFULL STRAIGHT TESTS = %{(total_num_tests_pass_hands['STRAIGHT']/total_num_tests_hands['STRAIGHT'])*100} ({total_num_tests_pass_hands['STRAIGHT']}/{total_num_tests_hands['STRAIGHT']})")
            print(f"AVERAGE STRAIGHT FLOP PREDICTION = {(total_flop/total_num_tests_pass_hands['STRAIGHT'])}")
            print(f"AVERAGE STRAIGHT TURN PREDICTION = {(total_turn/total_num_tests_pass_hands['STRAIGHT'])}")
            print('\n')          
        
        if total_num_tests_hands['FLUSH'] != 0:
            total_flop = 0
            for prediction in total_tests_predictions_flop['FLUSH']:
                total_flop+=prediction
            total_turn = 0
            for prediction in total_tests_predictions_turn['FLUSH']:
                total_turn+=prediction

            print(f"PERCENT SUCCESSFULL FLUSH TESTS = %{(total_num_tests_pass_hands['FLUSH']/total_num_tests_hands['FLUSH'])*100} ({total_num_tests_pass_hands['FLUSH']}/{total_num_tests_hands['FLUSH']})")
            print(f"AVERAGE FLUSH FLOP PREDICTION = {(total_flop/total_num_tests_pass_hands['FLUSH'])}")
            print(f"AVERAGE FLUSH TURN PREDICTION = {(total_turn/total_num_tests_pass_hands['FLUSH'])}")
            print('\n')         
        
        if total_num_tests_hands['FULL_HOUSE'] != 0:
            total_flop = 0
            for prediction in total_tests_predictions_flop['FULL_HOUSE']:
                total_flop+=prediction
            total_turn = 0
            for prediction in total_tests_predictions_turn['FULL_HOUSE']:
                total_turn+=prediction

            print(f"PERCENT SUCCESSFULL FULL_HOUSE TESTS = %{(total_num_tests_pass_hands['FULL_HOUSE']/total_num_tests_hands['FULL_HOUSE'])*100} ({total_num_tests_pass_hands['FULL_HOUSE']}/{total_num_tests_hands['FULL_HOUSE']})")
            print(f"AVERAGE FULL_HOUSE FLOP PREDICTION = {(total_flop/total_num_tests_pass_hands['FULL_HOUSE'])}")
            print(f"AVERAGE FULL_HOUSE TURN PREDICTION = {(total_turn/total_num_tests_pass_hands['FULL_HOUSE'])}")
            print('\n')         
        
        if total_num_tests_hands['QUADS'] != 0:
            total_flop = 0
            for prediction in total_tests_predictions_flop['QUADS']:
                total_flop+=prediction
            total_turn = 0
            for prediction in total_tests_predictions_turn['QUADS']:
                total_turn+=prediction

            print(f"PERCENT SUCCESSFULL QUADS TESTS = %{(total_num_tests_pass_hands['QUADS']/total_num_tests_hands['QUADS'])*100} ({total_num_tests_pass_hands['QUADS']}/{total_num_tests_hands['QUADS']})")
            print(f"AVERAGE QUADS FLOP PREDICTION = {(total_flop/total_num_tests_pass_hands['QUADS'])}")
            print(f"AVERAGE QUADS TURN PREDICTION = {(total_turn/total_num_tests_pass_hands['QUADS'])}")
            print('\n')         
        
        if total_num_tests_hands['STRAIGHT_FLUSH'] != 0:
            total_flop = 0
            for prediction in total_tests_predictions_flop['STRAIGHT_FLUSH']:
                total_flop+=prediction
            total_turn = 0
            for prediction in total_tests_predictions_turn['STRAIGHT_FLUSH']:
                total_turn+=prediction

            print(f"PERCENT SUCCESSFULL STRAIGHT_FLUSH TESTS = %{(total_num_tests_pass_hands['STRAIGHT_FLUSH']/total_num_tests_hands['STRAIGHT_FLUSH'])*100} ({total_num_tests_pass_hands['STRAIGHT_FLUSH']}/{total_num_tests_hands['STRAIGHT_FLUSH']})")
            print(f"AVERAGE STRAIGHT_FLUSH FLOP PREDICTION = {(total_flop/total_num_tests_pass_hands['STRAIGHT_FLUSH'])}")
            print(f"AVERAGE STRAIGHT_FLUSH TURN PREDICTION = {(total_turn/total_num_tests_pass_hands['STRAIGHT_FLUSH'])}")
            print('\n')         
        
        return

    def set_hand_evaluator(self):
        self.handevaluator = HandEvaluator()

    def test_royal_flush(self):
        self.royal_flush = [Card("S10", 'SPADES', '10'), Card("SJ", 'SPADES', 'J'), Card("SQ", 'SPADES', 'Q'), Card("SK", 'SPADES', 'K'), Card("SA", 'SPADES', 'A')]
        hand_value, best_five_card_combo = self.handevaluator.evaluate_7_card_hand(self.royal_flush)
        print(f"ROYAL FLUSH TEST HAND VALUE: {HAND_RANK_NAMES[hand_value[0]]}\nBEST_FIVE_CARD_HAND: {best_five_card_combo[0].id} {best_five_card_combo[1].id} {best_five_card_combo[2].id} {best_five_card_combo[3].id} {best_five_card_combo[4].id}")
        return hand_value[0]==9
        
    def test_flush_straight_flush(self):

        self.straight = [Card("SA", 'SPADES', 'A'), Card("C2", 'CLUBS', '2'), Card("H3", 'HEARTS', '3'), Card("H4", 'HEARTS', '4'), Card("D5", 'SPADES', '5')]
        self.straight_flush = []
        start = random.randint(2,5)
        for i in range(5):
            start += 1
            self.straight_flush.append(Card(f'D{start}','DIAMONDS',f"{start}"))

        sf_hand_value, sf_best_five_card_combo = self.handevaluator.evaluate_7_card_hand(self.straight_flush)
        print(f"STRAIGHT FLUSH TEST HAND VALUE: {HAND_RANK_NAMES[sf_hand_value[0]]}\nBEST_FIVE_CARD_HAND: {sf_best_five_card_combo[0].id} {sf_best_five_card_combo[1].id} {sf_best_five_card_combo[2].id} {sf_best_five_card_combo[3].id} {sf_best_five_card_combo[4].id}")
        s_hand_value, s_best_five_card_combo = self.handevaluator.evaluate_7_card_hand(self.straight)
        print(f"STRAIGHT TEST HAND VALUE: {HAND_RANK_NAMES[s_hand_value[0]]}\nBEST_FIVE_CARD_HAND: {s_best_five_card_combo[0].id} {s_best_five_card_combo[1].id} {s_best_five_card_combo[2].id} {s_best_five_card_combo[3].id} {s_best_five_card_combo[4].id}")
        return (sf_hand_value[0]==9 and s_hand_value[0]==5)

    def test_quad(self):

        deck = Deck()
        tests = []

        # run five hands with different kickers
        for i in range(5):
            hold = random.randint(2, 10)
            self.quad = [Card(f"D{hold}", 'DIAMONDS', f"{hold}"), Card(f"C{hold}", 'CLUBS', f"{hold}"), Card(f"H{hold}", 'HEARTS', f"{hold}"), Card(f"S{hold}", 'SPADES', f"{hold}")]
            hold_quad = self.quad.copy()
            random.shuffle(deck.cards)
            while len(hold_quad)<7:
                kicker = deck.cards.pop()
                if kicker in hold_quad:
                    deck.cards.insert(0, kicker)
                else:
                    hold_quad.append(kicker)

            hold_hand_value, hold_best_hand_value = self.handevaluator.evaluate_7_card_hand(hold_quad)
            tests.append([HAND_RANK_NAMES[hold_hand_value[0]], hold_best_hand_value, hold_hand_value[0]==8])

        passed = True
        for test in tests:
            if not test[2]:
                passed = False
            hand = ""
            for card in test[1]:
                hand += card.id + " "
            print(f"HAND: {test[0]}    PASS: {test[2]}    HAND: {hand} ")

        return passed

    def test_fuller_houses(self):

        suits = ['HEARTS', 'SPADES', 'CLUBS', 'DIAMONDS']
        low_values = ['2', '3', '4', '5', '6']
        high_values = ['8', '9', '10', 'J', 'Q']
        tests = []
        
        for i in range(5):

            random.shuffle(low_values)
            random.shuffle(high_values)

            self.full_house_big_three = []
            self.full_house_small_three = []

            for i in range (3):
                self.full_house_big_three.append(Card(f'{suits[i]}{high_values[0]}', suits[i],f"{high_values[0]}"))
                self.full_house_small_three.append(Card(f'{suits[i]}{low_values[0]}', suits[i],f"{low_values[0]}"))
            for i in range (2):
                self.full_house_small_three.append(Card(f'{suits[i]}{high_values[2]}', suits[i],f"{high_values[2]}"))
                self.full_house_big_three.append(Card(f'{suits[i]}{low_values[2]}', suits[i],f"{low_values[2]}"))
            
            self.full_house_big_three.append(Card('SA', 'SPADES', 'A')) # differnet cards not included in low/highvalues
            self.full_house_big_three.append(Card('HK', 'HEARTS', 'K')) # differnet cards not included in low/highvalues

            self.full_house_small_three.append(Card('SA', 'SPADES', 'A')) # differnet cards not included in low/highvalues
            self.full_house_small_three.append(Card('HK', 'HEARTS', 'K')) # differnet cards not included in low/highvalues

            small_hand_value, hold_best_hand_value = self.handevaluator.evaluate_7_card_hand(self.full_house_small_three)
            big_hand_value, hold_best_hand_value = self.handevaluator.evaluate_7_card_hand(self.full_house_big_three)

            tests.append({'small_hand':small_hand_value, 'big_hand':big_hand_value, 'pass': big_hand_value>small_hand_value})

        passed = True
        i=0
        print(f"FOR EACH TEST   (FULL HOUSE = {HAND_RANKS['FULL_HOUSE']}, TRIPLE CARD, PAIR CARD)    PASS TRUE if BIG HAND VALUE > SMALL HAND VALUE")
        for test in tests:
            if not test['pass']:
                passed = False
            hand = ""
            print(f"TEST {i}: BIG: {test['big_hand']}    SMALL: {test['small_hand']}    PASS: {test['pass']} ")
            i+=1

        return passed
    
    def test_flush(self):

        tests = []
        suits = ["SPADES", "HEARTS", "DIAMONDS", "CLUBS"]
        values: list = ['A', '2', '4', '5', '6', '8', '9', '10', 'Q', 'K'] #modifed for no straights

        for i in range(5):
            self.flush = []
            random.shuffle(suits)
            random.shuffle(values)
            seen_items = {}

            while len(self.flush)<7:
                if len(self.flush) < 5:
                    value = values[random.randint(0,9)]
                    if value not in seen_items:
                        self.flush.append(Card(f'{suits[0][0]}{value}', suits[0], value))
                        seen_items[value] = value
                elif 4 <= len(self.flush) < 7:
                    value = values[random.randint(0, 9)]
                    if value not in seen_items:
                        suit = suits[random.randint(1,3)][0]
                        self.flush.append(Card(f'{suit}{value}', suit, value))
                        seen_items[value] = value

            random.shuffle(self.flush)
            hand_value, best_five_card_hand = self.handevaluator.evaluate_7_card_hand(self.flush)

            hand = ""
            for card in self.flush:
                hand += card.id + " "

            best_hand = ""
            for card in best_five_card_hand:
                best_hand += card.id + " "

            tests.append([hand_value[0], hand, best_hand, hand_value[0]==HAND_RANKS['FLUSH']]) 

        passed = True
        i=0
        print(f"FOR EACH TEST IF HAND VALUE FLUSH IS ({HAND_RANKS["FLUSH"]})")
        for test in tests:
            if not test[2]:
                passed = False
            print(f"TEST {i}:    HAND VALUE: {test[0]}   HAND: {test[1]}    BEST HAND: {test[2]} PASS: {test[3]}")
            i+=1

        return passed

    def run_hand_evaluator_tests(self):
        print("\033c", end="", flush=True)
        print("- - - - - - - - - - - EVALUATOR TESTS RESULTS - - - - - - - - - - - -\n")
        print(f"\nTEST ROYAL FLUSH RESULT: {TEST_PASS[self.test_royal_flush()]}\n")
        print(f"\nTEST STRAIGHT FLUSH RESULT: {TEST_PASS[self.test_flush_straight_flush()]}\n")
        print(f"\nTEST QUAD RESULT: {TEST_PASS[self.test_quad()]}\n")
        print(f"\nTEST FULLER HOUSES: {TEST_PASS[self.test_fuller_houses()]}\n")
        print(f"\nTEST FLUSH: {TEST_PASS[self.test_flush()]}\n")

        print("- - - - - - - - - - - PREDICTION TESTS RESULTS - - - - - - - - - - - -\n")
        self.test_hand_prediction()

        # Basic hand rankings
#        - Royal flush detection (A♠ K♠ Q♠ J♠ 10♠) CHECK
#        - Straight flush vs regular flush (ensure straight flush ranks higher)
#        - Four of a kind with different kickers
#        - Full house comparison (higher three-of-a-kind wins)
#        - Flush with different high cards
#        - Straight with ace high vs ace low (A-2-3-4-5 vs 10-J-Q-K-A)
#        - Three of a kind with multiple kicker comparisons
#        - Two pair vs two pair (compare high pair, then low pair, then kicker)
#        - One pair with multiple kickers
#        - High card comparisons going down to 5th card
#        
#        # Edge cases
#        - Using best 5 cards from 7 available (player hand + community cards)
#        - Identical hands (true tie/chop scenario)
#        - Wheel straight (A-2-3-4-5)
#        - Multiple players with same hand type but different strengths
#        - All community card hands (board plays)#
#
#    def test_betting_flow():
#        # Pre-flop betting
#        - Action starting left of big blind pre-flop
#        - Minimum raise enforcement (must raise at least previous bet amount)
#        - All-in scenarios with less than minimum bet
#        
#        # Post-flop betting
#        - Action starting left of dealer on flop/turn/river
#        - Check-raise sequences
#        - Multiple raises in one betting round
#        - Cap on number of raises (if implemented)
#        
#        # Player actions
#        - Fold, check, call, bet, raise validation
#        - Can't check when there's a bet to call
#        - Can't call for more chips than player has
#        - All-in when betting all remaining chips
#        - String betting prevention (single action per turn)
#        
#        # Side pot scenarios
#        - Player all-in for less than current bet
#        - Multiple all-ins with different stack sizes
#        - Continuation of betting after all-in
#
#    def test_pot_resolution():
#        # Simple scenarios
#        - Winner takes entire pot (no side pots)
#        - Split pot between two players with identical hands
##        - Split pot with odd chip (who gets the extra chip?)
#       
#        # Side pot scenarios
#        - One all-in player, two players continue betting
##        - Multiple all-ins creating multiple side pots
#        - Player wins main pot but loses side pot
#        - Player wins side pot but loses main pot
#        - Player not eligible for side pot they didn't contribute to
#        
#        # Complex scenarios
#        - Three-way all-in with different stack sizes
#        - Multiple players split main pot, different winner for side pot
#        - Player folds after contributing to pot (chips stay in pot)
#        - All-in player loses but gets refund (if overpaying scenario exists)
#        
#        # Edge cases
#        - Verify correct chip distribution when multiple pots split
#        - Everyone folds to last player (wins without showdown)
#        - Rake/house fee deduction (if applicable)
#
#    def test_hand_prediction():
#        # Probability calculations
#        - Pre-flop: pocket aces vs random hand win rate (~85%)
#        - Pre-flop: suited connectors vs overpair
#        - Flop: flush draw probability (9 outs ≈ 35% by river)
#        - Flop: open-ended straight draw (8 outs ≈ 31.5% by river)
#        - Turn: drawing to specific outs with one card to come
#        
#        # Specific scenarios
#        - Two overcards on flop (6 outs)
#        - Gutshot straight draw (4 outs)
#       - Combo draws (flush + straight draw)
#        - Made hand vs drawing hand probabilities
#        - Set vs overpair after flop
#        # Edge cases
#        - Drawing dead (0% win probability)
#      - Already won/guaranteed win (100% probability)
#        - Redraw situations (winning hand can still be beaten)
#        - Counterfeit scenarios (river card kills your hand)
#        - Multiple opponents vs heads-up probability differences
#        
#        # Accuracy tests
#        - Monte Carlo simulation vs analytical calculation comparison
#        - Known probability scenarios (verify against poker odds charts)
#        - Reasonable computation time for probability estimation
##
if __name__=="__main__":
    tests()