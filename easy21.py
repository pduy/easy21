import numpy as np
from enum import Enum

Color = Enum('Color', 'red black')
Status = Enum('Status', 'fine bust')
COLORS = [Color.red, Color.black]


class Card:

    def __init__(self, val, color):
        self.color = color
        self.val = val if self.color == Color.black else -val


class BasePlayer(object):

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

    def __init__(self, cards):
        super(Player, self).__init__(cards)


class Dealer(BasePlayer):
    def __init__(self, cards):
        super(Dealer, self).__init__(cards)

    def act(self):
        while self.val() < 17:
            self.hit()


class State:
    def __init__(self, dealer, player):
        self.dealer = dealer
        self.player = player

    def take_action(self, player_action):
        if player_action == 'hit':
            self.player.hit()
        else:
            self.dealer.act()

        return State(self.dealer, self.player)


def draw(color=None):
    assert color is None or color in COLORS

    return Card(
        np.random.randint(1, 11),
        np.random.choice(COLORS, p=[1.0 / 3, 2.0 / 3])
        if color is None else color)


def step(state, player_action):
    return state.take_action(player_action)


def init_game():
    first_dealer = Dealer([draw(color=Color.black)])
    first_player = Player([draw(color=Color.black)])
    return State(first_dealer, first_player)


def info(state):
    print('dealer value = {0}'.format(state.dealer.val()))
    print('player value = {0}'.format(state.player.val()))


def main():
    state = init_game()

    info(state)

    while True:
        action = raw_input('enter your action:')
        state = step(state, action)
        info(state)


if __name__ == "__main__":
    main()
