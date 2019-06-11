import numpy as np

import easy21 as game
from learning import Sarsa
from table_lookup import SarsaLambdaPlayer


class LinearFunctionPlayer(SarsaLambdaPlayer):
    """
    The exploration strategy here is almost the same as Sarsa Lambda. The
    difference is that we use a Linear Regression to approximate the
    Action-Value Function (Q) instead of performing a Backward-Sarsa update.

    This strategy (using a supervised model to estimate the (action) value
    function) is effective in the case of very large or continuous action
    space. In our case the benefit is not that clear because the action choices
    are limited (only 2).
    """
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
