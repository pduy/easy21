from collections import namedtuple
from copy import deepcopy

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt

import easy21 as game
from easy21 import Color
from easy21 import Dealer
from easy21 import Environment
from easy21 import State
from easy21 import draw
from mpl_toolkits.mplot3d import Axes3D


def _print_dict(d):
    for k, v in d.items():
        print(f'{k}: {v}')


def init_game(algo='sarsa', slambda=0.5):
    dealer = Dealer([draw(color=Color.black)])
    environment = Environment(state=State(), dealer=dealer)

    if algo == 'sarsa':
        player = SarsaLambdaPlayer(cards=[draw(color=Color.black)],
                                   policy=Policy(),
                                   environment=environment,
                                   sarsa_lambda=slambda)
    elif algo == 'mcmc':
        player = MCPlayer(cards=[draw(color=Color.black)],
                          policy=Policy(),
                          environment=environment)
    else:
        player = LinearFunctionPlayer(cards=[draw(color=Color.black)],
                                      policy=Policy(),
                                      environment=environment,
                                      sarsa_lambda=slambda)

    environment.state.player_score = player.val()
    environment.state.dealer_score = dealer.val()
    return player, environment


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


class MCPlayer(Agent):
    def evaluate(self, episode):
        for i, (state, action) in enumerate(episode):
            if state.is_terminal:
                return

            self.count(state, action)

            alpha = 1.0 / self._count_states_actions.get((state, action), 0)
            q_value = self._compute_action_value_of(state, action)
            g_t = np.sum([s.reward() for s, a in episode[i:]])

            self._action_values[(state, action)] = \
                q_value + alpha * float(g_t - q_value)

    def sample_episode(self):
        player, environment = init_game(algo='mcmc')
        player._policy = deepcopy(self._policy)

        states_actions = []
        while player._environment.state is not None:
            action = player.choose_action()
            states_actions.append((player._environment.state, action))
            player._environment.state = player._environment.step(player, action)

        return states_actions

    def train(self, steps=100000):
        for i in range(steps):
            episode = self.sample_episode()
            self.evaluate(episode)
            self.update_eps_greedy_policy_for(episode)
        self.plot_value_function()

    def __repr__(self):
        return f'MCPlayer, gamma = {self._gamma}'


Sarsa = namedtuple('Sarsa', 'state action reward next_state next_action')


class SarsaLambdaPlayer(Agent):
    def __init__(self, cards, policy, environment, sarsa_lambda=0.0):
        super().__init__(cards, policy, environment)
        self._eligibility_trace = {}
        self.sarsa_lambda = sarsa_lambda

    def count(self, state, action=None):
        super().count(state, action)
        self._eligibility_trace[(state, action)] = \
            self._eligibility_trace.get((state, action), 0) + 1

    def update_action_values(self, sarsa: Sarsa):
        """ Backward Sarsa Lambda algorithm. """
        alpha = 1 / self._count_states_actions.get(
            (sarsa.state, sarsa.action), 0)
        q_value = self._compute_action_value_of(sarsa.state, sarsa.action)
        next_q_value = self._compute_action_value_of(sarsa.next_state,
                                                     sarsa.next_action)
        td_error = sarsa.reward + self._gamma * next_q_value - q_value

        for (s, a) in self._count_states_actions.keys():
            e_sa = self._eligibility_trace.get((s, a), 0)
            q_sa = self._compute_action_value_of(s, a)
            self._action_values[(s, a)] = q_sa + alpha * td_error * e_sa
            self._eligibility_trace[(s, a)] = \
                self._gamma * self.sarsa_lambda * e_sa

    def update_values_one_ep(self):
        player, environment = init_game()
        player._policy = deepcopy(self._policy)

        current_state = deepcopy(player._environment.state)
        action = player.choose_action()
        while not player._environment.state.is_terminal:
            self.count(current_state, action)

            player._environment.state = player._environment.step(player, action)
            reward = player._environment.state.reward()

            self.update_eps_greedy_policy(current_state)
            next_action = player.choose_action()
            self.update_action_values(
                Sarsa(current_state, action, reward, player._environment.state,
                      next_action))

            current_state = deepcopy(player._environment.state)
            action = next_action

    def train(self, steps=100000):
        for i in range(steps):
            self.update_values_one_ep()
        self.plot_value_function()

    def __repr__(self):
        return f'Sarsa Player, gamma = {self._gamma}, ' \
            f'lambda = {self.sarsa_lambda}'


class LinearFunctionPlayer(SarsaLambdaPlayer):
    def __init__(self, cards, policy, environment, sarsa_lambda=0.0):
        super().__init__(cards, policy, environment, sarsa_lambda)
        self._n_features = 36
        self._weights = np.random.uniform(low=-1, high=1, size=self._n_features)
        self._state_action_features = {}
        self._eligibility_trace = np.zeros(self._n_features)

    def update_action_values(self, sarsa: Sarsa):
        alpha = 0.01

        q_value = self._compute_action_value_of(sarsa.state, sarsa.action)
        next_q_value = self._compute_action_value_of(sarsa.next_state,
                                                     sarsa.next_action)

        td_error = sarsa.reward + self._gamma * next_q_value - q_value

        self._eligibility_trace *= self._gamma * self.sarsa_lambda
        self._eligibility_trace += self._get_features(sarsa.state, sarsa.action)
        self._weights += alpha * td_error * self._eligibility_trace
        self._action_values[(sarsa.state, sarsa.action)] = \
            self._compute_action_value_of(sarsa.state, sarsa.action)

    def _compute_action_value_of(self, state, action):
        features = self._get_features(state, action)
        return np.dot(features, self._weights)

    def _get_features(self, state, action):
        if (state, action) not in self._state_action_features.keys():
            self._state_action_features[(state, action)] = \
                _generate_features(state, action)
        return self._state_action_features[(state, action)]

    @staticmethod
    def _ep(count):
        """ Constant exploration. """
        return 0.05

    def count(self, state, action=None):
        """For this algorithm, count is not needed"""
        pass

    def __repr__(self):
        return f'LinearFunctionPlayer(gamma = {self._gamma}, ' \
            f'lambda = {self.sarsa_lambda})'


def _generate_features(state, action):
    dealer_ranges = [[1, 4], [4, 7], [7, 10]]
    player_ranges = [[1, 6], [4, 9], [7, 12], [10, 15], [13, 18], [16, 21]]

    if state.is_terminal:
        return np.zeros(
            len(dealer_ranges) * len(player_ranges) * len(game.ACTIONS))

    return np.array([
        int(dealer_range[0] <= state.dealer_score <= dealer_range[1]
            and player_range[0] <= state.player_score <= player_range[1]
            and action == a) for dealer_range in dealer_ranges
        for player_range in player_ranges for a in game.ACTIONS
    ])
