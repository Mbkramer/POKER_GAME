import random
from core.card import Card

#########
# DECK CLASS
#########
class Deck:
    def __init__(self):
            
            self.cards = []

            values = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
            suits = ['SPADES', 'DIAMONDS', 'HEARTS', 'CLUBS']

            for suit in suits:
                for value in values:
                    id = suit[0]+value 
                    self.cards.append(Card(id,suit,value))

    def shuffle(self):
        random.shuffle(self.cards)

    def print(self):
        print(len(self.cards))
        for card in self.cards:
            card.print()

    def deal(self):
         return self.cards.pop()