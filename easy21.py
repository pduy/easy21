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

    def __repr__(self):
        return 'Card({0}, {1})'.format(self.color, self.val)


class BasePlayer(object):
    def __init__(self, cards):
        self.cards = cards

    def val(self):
        return sum(c.val for c in self.cards)

    def hit(self):
        self.cards.append(draw())

    def __repr__(self):
        return f'Player({self.cards})'


class Player(BasePlayer):
    def __init__(self, cards, policy, environment):
        super(Player, self).__init__(cards)
        self.policy = policy
        self.environment = environment
        self.actions = [Action.hit, Action.stick]
        self.n_wins = 0

    def choose_action(self):
        if self.environment.state.is_terminal:
            return None
        return np.random.choice(self.actions,
                                p=self.policy.dist(self.environment.state))

    def sample_episode(self):
        player, environment = init_game()
        player.policy = deepcopy(self.policy)

        states = []
        while player.environment.state is not None:
            action = player.choose_action()
            states.append((player.environment.state, action))
            player.environment.state = player.environment.step(player, action)

        if states[len(states) - 1][0].reward() == 1:
            self.n_wins += 1

        return states

    def train(self):
        n_0 = 3
        steps = 10000

        def ep(a_state):
            return float(n_0) / (n_0 + count_states.get(a_state, 0))

        for i in range(100):
            print('step = {0}'.format(i))
            self.n_wins = 0
            count_states = {}
            count_states_actions = {}
            action_values = {}
            for _ in range(steps):
                episode = self.sample_episode()
                evaluate_episode(episode, count_states, count_states_actions,
                                 action_values)

            for s in count_states.keys():
                values = [action_values.get((s, a), 0) for a in self.actions]
                greedy_action = self.actions[np.argmax(values)]
                other_action = self.actions[1 - np.argmax(values)]
                self.policy.update(s, greedy_action, ep(s) / 2 + 1 - ep(s))
                self.policy.update(s, other_action, ep(s) / 2)

            print(f'Win rates = {self.n_wins / steps}')

        _print_dict(self.policy.state_action_probs)


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
    def from_players(cls, dealer_score, player_score, is_terminal):
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

    def __hash__(self):
        return hash((self.player_score, self.dealer_score))


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

    def __repr__(self):
        return 'Policy: ' + str(self.state_action_probs)


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


def _print_dict(d):
    for k, v in d.items():
        print(f'{k}: {v}')


def is_bust(score):
    return score <= -1 or score > 21


def draw(color=None):
    assert color is None or color in COLORS

    return Card(
        np.random.randint(1, 11),
        np.random.choice(COLORS, p=[1.0 / 3, 2.0 /
                                    3]) if color is None else color)


def init_game():
    dealer = Dealer([draw(color=Color.black)])
    environment = Environment(state=State(), dealer=dealer)
    player = Player(cards=[draw(color=Color.black)],
                    policy=Policy(),
                    environment=environment)
    environment.state.player_score = player.val()
    environment.state.dealer_score = dealer.val()

    return player, environment


def evaluate_episode(episode, count_states, count_states_actions, action_value):
    for i, (state, action) in enumerate(episode):
        if state.is_terminal:
            return

        count_states_actions[(state, action)] = count_states_actions.get(
            (state, action), 0) + 1
        count_states[state] = count_states.get(state, 0) + 1

        g_t = np.sum([s.reward() for s, a in episode[i:]])
        action_value[(state, action)] = action_value.get((state, action), 0) \
            + float(g_t - action_value.get((state, action), 0)) \
            / count_states_actions.get((state, action), 0)


def main():
    player, environment = init_game()
    player.train()


if __name__ == "__main__":
    main()
