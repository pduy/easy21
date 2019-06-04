from enum import Enum

import numpy as np

Color = Enum('Color', 'red black')
Action = Enum('Action', 'hit stick')
ACTIONS = [Action.hit, Action.stick]
COLORS = [Color.red, Color.black]


class Card:
    def __init__(self, val, color):
        self.color = color
        self.val = val if self.color == Color.black else -val

    def __repr__(self):
        return 'Card({0}, {1})'.format(self.color, self.val)


class BasePlayer:
    def __init__(self, cards):
        self.cards = cards

    def val(self):
        return sum(c.val for c in self.cards)

    def hit(self):
        self.cards.append(draw())

    def __repr__(self):
        return f'Player({self.cards})'


class Dealer(BasePlayer):
    def __init__(self, cards):
        super(Dealer, self).__init__(cards)

    def act(self):
        while self.val() < 17:
            self.hit()


class State:
    def __init__(self, is_terminal: bool = False):
        self.dealer_score = 0
        self.player_score = 0
        self.is_terminal = is_terminal

    @classmethod
    def from_players(cls, dealer_score, player_score, is_terminal=False):
        new_state = cls(is_terminal)
        new_state.dealer_score = dealer_score
        new_state.player_score = player_score
        return new_state

    def reward(self):
        if not self.is_terminal:
            return 0

        if is_bust(self.dealer_score) and not is_bust(self.player_score):
            return 1
        if not is_bust(self.dealer_score) and is_bust(self.player_score):
            return -1
        else:
            if self.player_score > self.dealer_score:
                return 1
            elif self.player_score < self.dealer_score:
                return -1
        return 0

    def __repr__(self):
        return 'State(player={0}, dealer={1}{2})' \
            .format(self.player_score, self.dealer_score,
                    ', terminal' if self.is_terminal else '')

    def __eq__(self, other):
        return self.dealer_score == other.dealer_score \
            and self.player_score == other.player_score

    def __ne__(self, other):
        return self.dealer_score != other.dealer_score \
            or self.player_score != other.player_score

    def __gt__(self, other):
        return repr(self) > repr(other)

    def __lt__(self, other):
        return repr(self) < repr(other)

    def __hash__(self):
        return hash((self.player_score, self.dealer_score))


class Environment:
    def __init__(self, state, dealer):
        self.state = state
        self.dealer = dealer

    def step(self, player, player_action):
        if self.state is None or self.state.is_terminal:
            return None

        is_terminal = self.state.is_terminal
        if player_action == Action.hit:
            player.hit()
            if is_bust(player.val()):
                is_terminal = True
        else:
            self.dealer.act()
            is_terminal = True

        return State.from_players(self.dealer.val(), player.val(), is_terminal)


def is_bust(score):
    return score <= -1 or score > 21


def draw(color=None):
    assert color is None or color in COLORS

    return Card(
        np.random.randint(1, 11),
        np.random.choice(COLORS, p=[1.0 / 3, 2.0 /
                                    3]) if color is None else color)


def init_basic_game():
    dealer = Dealer([draw(color=Color.black)])
    environment = Environment(state=State(), dealer=dealer)
    player = BasePlayer([draw(color=Color.black)])

    environment.state.player_score = player.val()
    environment.state.dealer_score = dealer.val()
    return player, environment
