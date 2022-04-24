import numpy as np
import math

from collections import defaultdict
from tqdm import tqdm

class EpsilonGreedyTabularQLearning():
    """
    An epsilon-greedy, tabular SARSA agent. "Tabular" means we are using
    discrete states. This agent represents the Q-table as a defaultdict.

    Attributes:
      env: A gym Env() object.
      gamma: float, Discount factor (0,1]
      decay_type: string, "exponential" or "formula"
      decay: float, either [0,1] (if "exponential") or larger
        (e.g., 100, 200, ..., if "formula")
      min_epsilon: float, The smallest value epsilon can decay to [0,1)
      epsilon: float
      min_alpha: float, The smallest value alpha can decay to (0,1)
      alpha: float
      return_vals: "score" to return the average of the final 100 trials,
        "progress" to return the returns and rolling 100-episode average for all
        episodes.
    """

    def __init__(self,
                 env, gamma, decay, decay_type, min_epsilon, epsilon, min_alpha, alpha,
                 return_vals):
        self.env = env
        self.gamma = gamma
        self.decay_type = decay_type
        self.decay = decay
        self.min_epsilon = min_epsilon
        self.epsilon = epsilon
        self.min_alpha = min_alpha
        self.alpha = alpha
        self.q = defaultdict(lambda: np.zeros(env.action_space.n))
        self.return_vals = return_vals

    def get_decayed_parameter(self, parameter, minimum, start, t):
        """
        Decays the parameter passed. Allows for two types of decay,
        though I only use exponential.

        eponential: Just multiplies the parameter by some (0,1] value each time
          (e.g., 0.999)
        formula: Uses the formula and the self.decay hyperparameter is larger
          (e.g., 150).

        Args:
          parameter: A float, epsilon or alpha.
          min: The minimum value we will allow the parameter to take.
          start: The maximum value we will allow the parameter to take.
          t: Current timestep.

        Returns:
          The decayed parameter.
        """

        if self.decay_type == "exponential":
            return parameter * self.decay

        elif self.decay_type == "formula":
            return max(
                minimum, min(start, 1. - math.log10((t + 1) / self.decay))
            )

    def choose_action(self, state, epsilon):
        """
        The epsilon-greedy action selection that we all know and love.
        """
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.q[state])
        return action

    def update_action_values(self,
                             alpha, state, action, reward, next_state):
        """
        Update action values according to SARSA (pg. 130 in the text)
        """
        self.q[state][action] += alpha * (reward + \
                                          self.gamma * max(self.q[next_state]) - self.q[state][action])

    def run_step(self,
                 state, action, epsilon, alpha):
        """
        Runs an episode.

        Args:
          state:
          action:
          epsilon:
          alpha:

        Returns:
          state:
          action: A
          reward: A scalar for the reward from the episode.
          done: A boolean indicating if the episode has terminated.
        """
        next_state, reward, done, _ = self.env.step(action)
        next_action = self.choose_action(next_state, epsilon)
        self.update_action_values(
            alpha, state, action, reward, next_state)
        action = next_action
        state = next_state
        return state, action, reward, done

    def run_episode(self,
                    epsilon, alpha, decay_alpha, decay_epsilon):
        """
        Runs a single episode of the cartpole task.

        Args:
          epsilon:
          alpha:
          decay_alpha:
          decay_epsilon:

        Returns:
          The return for the episode (scalar).
        """
        t = 0
        episode_return = 0
        state = self.env.reset()
        action = self.choose_action(state, epsilon)

        while True:
            if decay_alpha:
                alpha = self.get_decayed_parameter(
                    alpha, self.min_alpha, self.alpha, t
                )
            if decay_epsilon:
                epsilon = self.get_decayed_parameter(
                    epsilon, self.min_epsilon, self.epsilon, t
                )
            state, action, reward, done = self.run_step(
                state, action, alpha, epsilon)
            episode_return += reward
            t += 1
            if done:
                break
        # print(state[0])
        return episode_return

    def learn_task(self,
                   num_episodes, decay_alpha, decay_epsilon):
        """
        Runs num_episodes and learns the cartpole task.

        Args:
          num_episodes:
          decay_alpha: Bool, whether or not to decay alpha.
          decay_epsilon: Bool, whether or not to decay epsilon.

        Returns:
          if return_vals = "score", returns the average return for the last
            100 episodes.
          if return_vals = "progress", returns the return and rolling 100-episode
            average return for all episodes.
        """
        returns = []
        rolling_avg_100 = []
        alpha = self.alpha
        epsilon = self.epsilon

        for episode in tqdm(range(1, num_episodes)):

            episode_return = self.run_episode(epsilon, alpha, decay_alpha, decay_epsilon)
            returns.append(episode_return)

            if episode > 5:
                rolling_avg_100.append(np.mean(returns[-5:]))

        if self.return_vals == "score":
            score = np.mean(returns[-5:])
            return score
        elif self.return_vals == "progress":
            return returns, rolling_avg_100

    def play_step(self,
                 state, action, epsilon, alpha):
        """
        Runs an episode.

        Args:
          state:
          action:
          epsilon:
          alpha:

        Returns:
          state:
          action: A
          reward: A scalar for the reward from the episode.
          done: A boolean indicating if the episode has terminated.
        """
        next_state, reward, done, _ = self.env.step(action)
        next_action = self.choose_action(next_state, epsilon)
        action = next_action
        state = next_state
        return state, action, reward, done

    def play(self,
                    epsilon, alpha, decay_alpha, decay_epsilon):
        """
        Runs a single episode of the cartpole task.

        Args:
          epsilon:
          alpha:
          decay_alpha:
          decay_epsilon:
          discretize_func:
          **discretize_kwargs:

        Returns:
          The return for the episode (scalar).
        """
        t = 0
        episode_return = 0
        state = self.env.reset()
        action = self.choose_action(state, epsilon)

        while True:
            self.env.clock.tick(30)
            if decay_alpha:
                alpha = self.get_decayed_parameter(
                    alpha, self.min_alpha, self.alpha, t
                )
            if decay_epsilon:
                epsilon = self.get_decayed_parameter(
                    epsilon, self.min_epsilon, self.epsilon, t
                )
            state, action, reward, done = self.play_step(
                state, action, alpha, epsilon)
            self.env.render()
            episode_return += reward
            t += 1
            if done:
                break
        # print(state[0])
        return episode_return