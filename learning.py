from collections import namedtuple

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt

import easy21 as game
from easy21 import State

from mpl_toolkits.mplot3d import Axes3D


def _print_dict(d):
    for k, v in d.items():
        print(f'{k}: {v}')


class Policy:
    def __init__(self):
        self.state_action_probs = {}

    def update(self, state, action, prob):
        self.state_action_probs[(state, action)] = prob

    def dist(self, state):
        return [self.prob(state, a) for a in game.ACTIONS]

    def prob(self, state, action):
        return self.state_action_probs.get((state, action), 0.5)


class Agent(game.BasePlayer):
    def __init__(self, cards, policy, environment, gamma=1):
        super(Agent, self).__init__(cards)
        self._policy = policy
        self._environment = environment
        self._gamma = gamma
        self._action_values = {}
        self._count_states = {}
        self._count_states_actions = {}

    @classmethod
    def reset(cls, **kwargs):
        raise NotImplementedError

    def count(self, state, action=None):
        if action is not None:
            self._count_states_actions[(state, action)] = \
                self._count_states_actions.get((state, action), 0) + 1
        self._count_states[state] = self._count_states.get(state, 0) + 1

    def choose_action(self):
        if self._environment.state.is_terminal:
            return None

        return np.random.choice(game.ACTIONS,
                                p=self._policy.dist(self._environment.state))

    def update_eps_greedy_policy_for(self, episode):
        for s, _ in episode:
            if s.is_terminal:
                break
            self.update_eps_greedy_policy(s)

    def update_eps_greedy_policy(self, state):
        count_s = self._count_states.get(state, 0)
        values = [self._compute_action_value_of(state, a) for a in game.ACTIONS]
        greedy_action = game.ACTIONS[np.argmax(values)]
        other_action = game.ACTIONS[1 - np.argmax(values)]
        self._policy.update(state, greedy_action,
                            self._ep(count_s) / 2 + 1 - self._ep(count_s))
        self._policy.update(state, other_action, self._ep(count_s) / 2)

    @staticmethod
    def _ep(count):
        n_0 = 1
        return float(n_0) / (n_0 + count)

    def get_value_fn_table(self):
        states = np.unique([s for s, a in self._action_values.keys()])
        return {state: self._compute_value_of(state) for state in states}

    def plot_value_function(self):
        dealer_scores = np.arange(12)
        player_scores = np.arange(22)
        dealer_scores, player_scores = np.meshgrid(dealer_scores, player_scores)
        values = np.array([
            self._compute_value_of(State.from_players(d, p))
            for d, p in zip(np.ravel(dealer_scores), np.ravel(player_scores))
        ])
        values = np.array(values)
        values = values.reshape(dealer_scores.shape)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('Value function heat map w.r.t dealer and player status')
        ax.plot_surface(X=dealer_scores,
                        Y=player_scores,
                        Z=values,
                        cmap=cm.coolwarm,
                        linewidth=0)
        ax.set_xlabel('Dealer score')
        ax.set_ylabel('Player score')
        ax.set_zlabel('Value')
        plt.show()

    def _compute_value_of(self, state):
        """ Value function V of a state is the expected Action Value Function
        Q of all actions from that state.
        V = sum_over_a (policy(a) * Q(a))
        """
        values_each_action = [
            self._policy.prob(state, a) *
            self._compute_action_value_of(state, a) for a in game.ACTIONS
        ]
        return np.sum(values_each_action)

    def _compute_action_value_of(self, state, action):
        return self._action_values.get((state, action), 0)


Sarsa = namedtuple('Sarsa', 'state action reward next_state next_action')
