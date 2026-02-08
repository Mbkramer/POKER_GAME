#########
# CARD CLASS
#########

# ---------- Card and Hand ranking ----------
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

HAND_RANKS = {
    "HIGH": 1,
    "PAIR": 2,
    "TWO_PAIR": 3,
    "TRIPLES": 4,
    "STRAIGHT": 5,
    "FLUSH": 6,
    "FULL_HOUSE": 7,
    "QUADS": 8,
    "STRAIGHT_FLUSH": 9
}

HAND_RANK_NAMES = {
    HAND_RANKS["HIGH"]: "High Card",
    HAND_RANKS["PAIR"]: "Pair",
    HAND_RANKS["TWO_PAIR"]: "Two Pair",
    HAND_RANKS["TRIPLES"]: "Three of a Kind",
    HAND_RANKS["STRAIGHT"]: "Straight",
    HAND_RANKS["FLUSH"]: "Flush",
    HAND_RANKS["FULL_HOUSE"]: "Full House",
    HAND_RANKS["QUADS"]: "Four of a Kind",
    HAND_RANKS["STRAIGHT_FLUSH"]: "Straight Flush"
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