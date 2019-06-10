from copy import deepcopy

import numpy as np

from easy21 import Dealer, draw, Color, Environment, State
from learning import Agent, Sarsa


class MCPlayer(Agent):
    @classmethod
    def reset(cls, **kwargs):
        dealer = Dealer([draw(color=Color.black)])
        environment = Environment(state=State(), dealer=dealer)

        player = cls(cards=[draw(color=Color.black)],
                     policy=kwargs['policy'],
                     environment=environment)

        environment.state.player_score = player.val()
        environment.state.dealer_score = dealer.val()
        return player, environment

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
        player, environment = self.reset(policy=self._policy)

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


class SarsaLambdaPlayer(Agent):
    def __init__(self, cards, policy, environment, sarsa_lambda=0.0):
        super().__init__(cards, policy, environment)
        self._eligibility_trace = {}
        self.sarsa_lambda = sarsa_lambda

    @classmethod
    def reset(cls, **kwargs):
        dealer = Dealer([draw(color=Color.black)])
        environment = Environment(state=State(), dealer=dealer)

        player = cls(cards=[draw(color=Color.black)],
                     policy=kwargs['policy'],
                     environment=environment,
                     sarsa_lambda=kwargs['sarsa_lambda'])

        environment.state.player_score = player.val()
        environment.state.dealer_score = dealer.val()
        return player, environment

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
        player, environment = self.reset(policy=self._policy,
                                         sarsa_lambda=self.sarsa_lambda)

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
