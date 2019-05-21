import numpy as np
from enum import Enum

Color = Enum('Color', 'red black')
Status = Enum('Status', 'fine bust')
Action = Enum('Action', 'hit stick')
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
        if 1 < self.val() <= 21:
            return Status.fine
        return Status.bust

    def is_bust(self):
        return self.status() == Status.bust

    def hit(self):
        self.cards.append(draw())
        if self.status == Status.bust:
            print(repr(self) + ' lost!')


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

    def __init__(self, dealer, player, status='active'):
        self.dealer = dealer
        self.player = player
        self.status = status
        if self.dealer.is_bust() or self.player.is_bust():
            self.status = 'terminal'

    def take_action(self, player_action):
        status = self.status
        if player_action == Action.hit:
            self.player.hit()
        else:
            self.dealer.act()
            status = 'terminal'
        return State(self.dealer, self.player, status)

    def reward(self):
        if self.dealer.is_bust() and not self.player.is_bust():
            return 1
        if not self.dealer.is_bust() and self.player.is_bust():
            return -1
        else:
            if self.player.val() > self.dealer.val():
                return 1
            elif self.player.val() < self.dealer.val():
                return -1
        return 0

    def __repr__(self):
        return 'State(player val = {0}, dealer val = {1}, status = {2})'.format(
            self.player.val(), self.dealer.val(), self.status)


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


def run_episode():
    state = init_game()
    states = [(state)]
    reward = None
    while (True):
        action = np.random.choice([Action.hit, Action.stick])
        next_state = step(state, action)
        states.append((action, next_state))
        if next_state.status == 'terminal':
            reward = next_state.reward()
            break

    return states, reward


def main():
    for i in range(10):
        print(run_episode())


if __name__ == "__main__":
    main()
