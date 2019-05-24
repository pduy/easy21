import numpy as np
from enum import Enum
from copy import deepcopy

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

    def __init__(self, cards, policy):
        super(Player, self).__init__(cards)
        self.policy = policy

    def choose_action(self, state):
        return np.random.choice([Action.hit, Action.stick],
                p=self.policy.dist(state))



class Dealer(BasePlayer):

    def __init__(self, cards):
        super(Dealer, self).__init__(cards)

    def act(self):
        while self.val() < 17:
            self.hit()


class State:

    def __init__(self, dealer_score, player_score, status='active'):
        self.dealer_score = dealer_score
        self.player_score = player_score
        self.status = status
        if is_bust(dealer_score) or is_bust(player_score):
            self.status = 'terminal'

    def reward(self):
        if self.status is not 'terminal':
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
        return 'State(player val = {0}, dealer val = {1}, status = {2})'.format(
            self.player_score, self.dealer_score, self.status)


class Policy:

    def __init__(self):
        self.state_action_probs = {}

    def update(self, state, action, prob):
        self.state_action_probs[(state, action)] = prob

    def dist(self, state):
        return [
            self.state_action_probs.get((state, a), 0.5)
            for a in [Action.hit, Action.stick]
        ]


def is_bust(score):
    return score <= -1 or score > 21


def draw(color=None):
    assert color is None or color in COLORS

    return Card(
        np.random.randint(1, 11),
        np.random.choice(COLORS, p=[1.0 / 3, 2.0 / 3])
        if color is None else color)


def step(dealer, player, state, player_action):
    if state is None or state.status == 'terminal':
        return None

    status = state.status

    if player_action == Action.hit:
        player.hit()
    else:
        dealer.act()
        status = 'terminal'

    return State(dealer.val(), player.val(), status)


def init_game():
    dealer = Dealer([draw(color=Color.black)])
    player = Player([draw(color=Color.black)], Policy())
    return dealer, player, State(dealer.val(), player.val())


def sample_episode(dealer, player, state):
    state = deepcopy(state)

    states = []
    while (True):
        action = player.choose_action(state)
        state = step(dealer, player, state, action)
        if state is None:
            break
        states.append(state)

    return states


def compute_return(episode):
    return sum([s.reward() for s in episode])


def value_fn(dealer, player, state, action):
    next_state = step(dealer, player, state, action)
    episodes = [sample_episode(dealer, player, next_state) for _ in range(1000)]
    returns = [compute_return(e) for e in episodes]
    return np.mean(returns)


def optimize(dealer, player, state):
    actions = [Action.hit, Action.stick]
    done = False
    state = deepcopy(state)
    N0 = 100
    NS = {state: 1}

    def ep():
        return N0 / (N0 + NS.get(state, 0))

    while not done:
        hit_or_stick_vals = [value_fn(dealer, player, state, a) for a in actions]
        max_action = actions[np.argmax(hit_or_stick_vals)]
        other_action = actions[2 / (np.argmax(hit_or_stick_vals) + 1) - 1]

        print('max action = {0}'.format(max_action))
        print('other action = {0}'.format(other_action))

        player.policy.update(state, max_action, ep() / 2 + 1 - ep())
        player.policy.update(state, other_action, ep() / 2)
        state = step(dealer, player, state, max_action)

        done = state is None
        NS[state] = NS.get(state, 0) + 1

        print(NS)
        print(player.policy.state_action_probs)


def main():
    dealer, player, state = init_game()
    # print(value_fn(dealer, player, state, Action.hit))
    # print(value_fn(dealer, player, state, Action.stick))
    optimize(dealer, player, state)


if __name__ == "__main__":
    main()
