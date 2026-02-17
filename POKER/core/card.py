#########
# CARD CLASS
#########

# ---------- Card ranking ----------
CARD_VALUE = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
    '7': 7, '8': 8, '9': 9, '10': 10,
    'J': 11, 'Q': 12, 'K': 13, 'A': 14
}

SUIT_VALUE = {

    'CLUBS': 0,
    'DIAMONDS': 1,
    'HEARTS': 2,
    'SPADES': 3
}

VALUE_NAME = {
    11: "Jack",
    12: "Queen",
    13: "King",
    14: "Ace"
}

class Card: 
    def __init__(self, id, suit, value):
        self.id = id
        self.suit = suit
        self.value = value

    def __hash__(self):
        return hash((CARD_VALUE[self.value], SUIT_VALUE[self.suit]))

    # Define equality based on name and grade
    def __eq__(self, other):
        if not isinstance(other, Card):
            return NotImplemented
        return (CARD_VALUE[self.value] == CARD_VALUE[other.value]) and (SUIT_VALUE[self.suit] == SUIT_VALUE[other.suit])

    # Define less than based on grade
    def __lt__(self, other):
        if not isinstance(other, Card):
            return NotImplemented
        
        # Primary comparison: value 
        if CARD_VALUE[self.value] < CARD_VALUE[other.value]:
            return True
        elif CARD_VALUE[self.value] > CARD_VALUE[other.value]:
            return False
        else:
            # Secondary comparison: name (if grades are equal)
            return SUIT_VALUE[self.suit] < SUIT_VALUE[other.suit]

    def print(self):
        print(f"{self.value} of {self.suit} ({self.id})")

    def get_card_string(self):
        return f"{self.value} of {self.suit}"