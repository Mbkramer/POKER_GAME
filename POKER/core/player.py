from core.card import Card

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
        self.folded = False
        self.all_in = False
        self.touched = False
        self.hand_value = 0

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
    
    def __repr__(self) -> str:
        return (
            f"Player(name={self.name}, "
            f"wallet={self.wallet}, "
            f"bet={self.current_bet}, "
            f"folded={self.folded}, "
            f"all_in={self.all_in})"
        )
    