from collections import namedtuple
from copy import deepcopy

import numpy as np

import easy21 as game
from easy21 import Color
from easy21 import Dealer
from easy21 import Environment
from easy21 import State
from easy21 import draw

Sarsa = namedtuple('Sarsa', 'state action reward next_state next_action')


def _print_dict(d):
    for k, v in d.items():
        print(f'{k}: {v}')


def ep(count):
    n_0 = 10
    return float(n_0) / (n_0 + count)


def init_game(algo='sarsa'):
    dealer = Dealer([draw(color=Color.black)])
    environment = Environment(state=State(), dealer=dealer)

    if algo == 'sarsa':
        player = SarsaPlayer(cards=[draw(color=Color.black)],
                             policy=Policy(),
                             environment=environment,
                             sarsa_lambda=0.9)
    else:
        player = MCPlayer(cards=[draw(color=Color.black)],
                          policy=Policy(),
                          environment=environment)

    environment.state.player_score = player.val()
    environment.state.dealer_score = dealer.val()
    return player, environment


class Agent(game.BasePlayer):
    def __init__(self, cards, policy, environment, gamma=1):
        super(Agent, self).__init__(cards)
        self.policy = policy
        self.environment = environment
        self.actions = [game.Action.hit, game.Action.stick]
        self.gamma = gamma
        self.n_wins = 0
        self.action_values = {}
        self.count_states = {}
        self.count_states_actions = {}

    def count(self, state, action=None):
        if action is not None:
            self.count_states_actions[(state, action)] = \
                self.count_states_actions.get((state, action), 0) + 1
        self.count_states[state] = self.count_states.get(state, 0) + 1

    def choose_action(self):
        if self.environment.state.is_terminal:
            return None

        return np.random.choice(self.actions,
                                p=self.policy.dist(self.environment.state))

    def greedy_update_policy(self, state):
        count_s = self.count_states.get(state, 0)
        values = [self.action_values.get((state, a), 0) for a in self.actions]
        greedy_action_index = np.argmax(values)
        greedy_action = self.actions[greedy_action_index]
        other_action = self.actions[1 - np.argmax(values)]
        self.policy.update(state, greedy_action,
                           ep(count_s) / 2 + 1 - ep(count_s))
        self.policy.update(state, other_action, ep(count_s) / 2)


class MCPlayer(Agent):
    def evaluate_episode(self, episode):
        for i, (state, action) in enumerate(episode):
            if state.is_terminal:
                return

            self.count(state, action)
            alpha = 1 / self.count_states_actions.get((state, action), 0)

            q_value = self.action_values.get((state, action), 0)
            g_t = np.sum([s.reward() for s, a in episode[i:]])
            self.action_values[(state, action)] = \
                q_value + alpha * float(g_t - q_value)

    def sample_episode(self):
        player, environment = init_game(algo='mcmc')
        player.policy = deepcopy(self.policy)

        states_actions = []
        while player.environment.state is not None:
            action = player.choose_action()
            states_actions.append((player.environment.state, action))
            player.environment.state = player.environment.step(player, action)

        if states_actions[len(states_actions) - 1][0].reward() == 1:
            self.n_wins += 1

        return states_actions

    def train(self):
        steps = 1000

        self.n_wins = 0
        for _ in range(steps):
            episode = self.sample_episode()
            self.evaluate_episode(episode)

            for s, _ in episode:
                if s.is_terminal:
                    continue
                self.greedy_update_policy(s)

        _print_dict(self.policy.state_action_probs)


class SarsaPlayer(Agent):
    def __init__(self, cards, policy, environment, sarsa_lambda=0.0):
        super().__init__(cards, policy, environment)
        self.eligibility_trace = {}
        self.sarsa_lambda = sarsa_lambda

    def count(self, state, action=None):
        super().count(state, action)
        self.eligibility_trace[(state, action)] = \
            self.eligibility_trace.get((state, action), 0) + 1

    def update_action_values(self, sarsa: Sarsa):
        alpha = 1 / self.count_states_actions.get(
            (sarsa.state, sarsa.action), 0)
        q_value = self.action_values.get((sarsa.state, sarsa.action), 0)
        next_q_value = self.action_values.get(
            (sarsa.next_state, sarsa.next_action), 0)
        td_error = sarsa.reward + self.gamma * next_q_value - q_value
        for (s, a) in self.count_states_actions.keys():
            e_sa = self.eligibility_trace.get((s, a), 0)
            q_sa = self.action_values.get((s, a), 0)
            self.action_values[(s, a)] = q_sa + alpha * td_error * e_sa
            self.eligibility_trace[(s, a)] = \
                self.gamma * self.sarsa_lambda * e_sa

    def update_values_one_ep(self):
        player, environment = init_game()
        player.policy = deepcopy(self.policy)

        current_state = deepcopy(player.environment.state)
        action = player.choose_action()
        while not player.environment.state.is_terminal:
            self.count(current_state, action)

            player.environment.state = player.environment.step(player, action)
            reward = player.environment.state.reward()

            self.greedy_update_policy(current_state)
            next_action = player.choose_action()
            self.update_action_values(
                Sarsa(current_state, action, reward, player.environment.state,
                      next_action))

            current_state = deepcopy(player.environment.state)
            action = next_action

    def train(self):
        steps = 1000

        self.n_wins = 0
        for _ in range(steps):
            self.update_values_one_ep()

        # _print_dict(self.action_values)


class Policy:
    def __init__(self):
        self.state_action_probs = {}

    def update(self, state, action, prob):
        self.state_action_probs[(state, action)] = prob

    def dist(self, state):
        return [
            self.state_action_probs.get((state, a), 0.5)
            for a in [game.Action.hit, game.Action.stick]
        ]
