import numpy as np
from enum import Enum

Color = Enum('Color', 'red black')
Status = Enum('Status', 'fine bust')
COLORS = [Color.red, Color.black]


class Card:

    def __init__(self, val, color):
        self.color = color
        self.val = val if self.color == Color.black else -val


class BasePlayer:

    def __init__(self, cards):
        self.cards = cards

    def val(self):
        return sum(c.val for c in self.cards)

    def status(self):
        if 1 <= self.val() <= 21:
            return Status.fine
        return Status.bust

    def hit(self):
        self.cards.append(draw())


class Player(BasePlayer):

    def __init__(self):
        super(Player, self).__init__()


class Dealer(BasePlayer):

    def __init__(self):
        super(Dealer, self).__init__()

    def act(self):
        if self.val() < 17:
            self.hit()


# class State:
#     def __init__(self, dealer, player):


def draw(color=None):
    assert color is None or color in COLORS

    return Card(
        np.random.randint(1, 11),
        np.random.choice(COLORS, p=[1.0 / 3, 2.0 / 3])
        if color is None else color)


def step(state, action):
    return ''
