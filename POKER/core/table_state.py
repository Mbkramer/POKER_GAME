# core/table_state.py
from typing import List
from core.player import Player
from core.card import Card
from core.deck import Deck

class TableState:
    def __init__(self, num_players, buy_in, wallet):

        self.num_players = num_players
        self.buy_in = buy_in
        self.wallet = wallet

        self.players = []

        for player in range(num_players):
            self.players.append(Player(player, wallet, buy_in))

        self.community_cards: List[Card] = []
        self.burn_deck: List[Card] = []
        self.deck: List[Card] = []
        
        self.pot = 0
        self.current_bet = 0
        
        self.dealer_index = 0
        self.small_blind = 0
        self.big_blind = 1

    def active_players(self) -> List[Player]:
        return [p for p in self.players if not p.folded]

    def reset_bets(self):
        for p in self.players:
            p.bet = 0
    
    def print(self):
        print(f"num_players:{self.num_players}\n")
        try:
            for player in self.players:
                print(f"Player ID {player.id}\nwallet: {self.wallet}\nbuy-in: {self.buy_in}\n")
        except:
            print("NO PLAYERS\n")
