import gym
import pygame
import numpy as np
from gym import spaces

window_width, window_height = 1000, 500
rotation_max, acceleration_max = 0.08, 0.5


class ETEnv(gym.Env):
    def __init__(self):
        low = np.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0
            ],
            dtype=np.int32,
        )
        high = np.array(
            [
                60,
                60,
                60,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1
            ],
            dtype=np.int32,
        )

        self.observation_space = spaces.Box(low, high, dtype=np.int32)
        self.state = None

        self.action_space = spaces.Discrete(13)

        self.num_towers = 3
        self.scores = [15, 20, 30]
        self.towers_type = [0, 0, 1]
        self.edges = {(0, 1): 1, (0, 2): 1, (1, 0): 0, (1, 2): 0, (2, 0): 1, (2, 1): 1}
        self.exit_edges = [2, 0, 2]
        self.x = [window_width / 4, 3*window_width / 4, window_width / 2]
        self.y = [window_height / 2, window_height / 2, window_height / 4]
        self.prev_sum_score = np.sum(self.scores[:2])

    def reset(self):
        # reset the environment to initial state
        self.scores = [15, 20, 30]
        self.towers_type = [0, 0, 1]
        self.edges = {(0, 1): 1, (0, 2): 1, (1, 0): 0, (1, 2): 0, (2, 0): 1, (2, 1): 1}
        self.exit_edges = [2, 0, 2]
        self.prev_sum_score = np.sum(self.scores[:2])

        # Building current state using scores and tower_types and edges
        self.state = []
        for score in self.scores:
            self.state += [int(score)]
        self.state += self.towers_type
        for edge in self.edges:
            self.state += [self.edges[edge]]

        return tuple(self.state)

    def step(self, action):

        # calculating the value of current state to be used in calculating reward
        prev_reward = 0
        for i in range(self.num_towers):
            prev_reward += (1 - 2 * self.towers_type[i]) * self.scores[i]

        dictionary = {0:(0,1), 1:(0,2), 2:(1,0), 3:(1,2), 4:(2,0), 5:(2,1), 6:(0,1), 7:(0,2), 8:(1,0), 9:(1,2), 10:(2,0), 11:(2,1)}
        if action <= 5:
            # Create edge based on chosen action
            edge = dictionary[action]
            if int(self.scores[edge[0]]/10) - self.exit_edges[edge[0]] > 0 and self.towers_type[edge[0]] == 0:
                if self.edges[edge] == 0:
                    self.exit_edges[edge[0]] += 1
                self.edges[edge] = 1
                if self.towers_type[edge[0]] == self.towers_type[edge[1]]:
                    if self.edges[(edge[1],edge[0])] == 1:
                        self.exit_edges[edge[1]] -= 1
                    self.edges[(edge[1],edge[0])] = 0

        elif action <= 11:
            # remove edge based on chosen action
            edge = dictionary[action]
            if self.towers_type[edge[0]] == 0:
                if self.edges[edge] == 1:
                    self.exit_edges[edge[0]] -= 1
                self.edges[edge] = 0

        # updating the score of towers
        r = 0
        for edge in self.edges.keys():
            if self.edges[edge] == 1:
                power_0 = self.scores[edge[0]] / self.exit_edges[edge[0]]
                if self.towers_type[edge[0]] == self.towers_type[edge[1]]:

                    # if allowed number of exiting edges increases, consider 10 reward
                    if int(self.scores[edge[1]]/10) <= 2 and self.towers_type[edge[1]] == 0:
                        if int((self.scores[edge[1]]+power_0/(60*30))/10) > int(self.scores[edge[1]]/10):
                            r += 10

                    self.scores[edge[1]] += power_0 / (60 * 30)
                    self.scores[edge[1]] = min(self.scores[edge[1]], 60)
                elif self.edges[(edge[1],edge[0])] == 1:
                    power_1 = self.scores[edge[1]] / self.exit_edges[edge[1]]
                    if power_1 <= power_0:

                        # if allowed number of exiting edges decreases, consider -10 reward
                        if int(self.scores[edge[1]] / 10) <= 3 and self.towers_type[edge[1]] == 0:
                            if int((self.scores[edge[1]] - (power_0-power_1) / (60 * 30)) / 10) < int(self.scores[edge[1]] / 10):
                                r -= 10

                        self.scores[edge[1]] -= (power_0-power_1) / (60 * 30)
                else:

                    # if allowed number of exiting edges decreases, consider -10 reward
                    if int(self.scores[edge[1]]/10) <= 3 and self.towers_type[edge[1]] == 0:
                        if int((self.scores[edge[1]]-power_0/(60*30))/10) < int(self.scores[edge[1]]/10):
                            r -= 10

                    self.scores[edge[1]] -= power_0 / (60 * 30)

        # changing the owner of towers which their score has become negative
        for i in range(self.num_towers):
            if self.scores[i] < 0:
                self.scores[i] = 0
                self.towers_type[i] = 1 - self.towers_type[i]
                self.exit_edges[i] = 0
                for edge in self.edges.keys():
                    if edge[0] == i:
                        self.edges[edge] = 0

        # Building current state using scores and tower_types and edges
        self.state = []
        for score in self.scores:
            self.state += [int(score)]
        self.state += self.towers_type
        for edge in self.edges:
            self.state += [self.edges[edge]]

        reward = 0
        for i in range(self.num_towers):
            reward += (1-2*self.towers_type[i])*self.scores[i]

        # The game is over when the type of all the towers is similar
        done = True
        for i in range(self.num_towers):
            if self.towers_type[i] != self.towers_type[0]:
                done = False

        # Winning has 1000 and losing has -1000 reward
        if done and self.towers_type[0] == 0:
            r += 1000
        if done and self.towers_type[0] == 1:
            r -= 1000

        return tuple(self.state), (reward-prev_reward)*100+r, done, {}

    def init_render(self):
        pygame.init()
        self.window = pygame.display.set_mode((window_width, window_height))
        self.clock = pygame.time.Clock()
        print(self.observation_space)

    def render(self):

        colors = [(0, 200, 200), (200, 0, 0)]
        self.window.fill((0, 0, 0))

        screen = pygame.display.set_mode((window_width, window_height))
        pygame.font.init()
        my_font = pygame.font.SysFont('Comic Sans MS', 10)
        # Display score of towers
        for i in range(self.num_towers):
            text_surface = my_font.render(str(int(self.scores[i])), False, (255, 255, 255))
            screen.blit(text_surface, (int(self.x[i]) - 5, int(self.y[i]) - 25))

        # Display towers
        for i in range(self.num_towers):
            pygame.draw.circle(self.window, colors[self.towers_type[i]], (int(self.x[i]), int(self.y[i])), 6)

        # Display edges
        for edge in self.edges.keys():
            if self.edges[edge] == 1:
                if self.towers_type[edge[0]] != self.towers_type[edge[1]] and self.edges[edge[1],edge[0]]==1:
                    p1 = (0.9*self.x[edge[0]] + 0.1*(self.x[edge[0]]+self.x[edge[1]])/2, 0.9*self.y[edge[0]] + 0.1*(self.y[edge[0]]+self.y[edge[1]])/2)
                    p2 = ((self.x[edge[0]]+self.x[edge[1]])/2, (self.y[edge[0]]+self.y[edge[1]])/2)
                    pygame.draw.line(self.window, colors[self.towers_type[edge[0]]], p1, p2, 2)
                else:
                    p1 = (0.9*self.x[edge[0]]+ 0.1*self.x[edge[1]], 0.9*self.y[edge[0]] + 0.1*self.y[edge[1]])
                    p2 = (self.x[edge[1]], self.y[edge[1]])
                    pygame.draw.line(self.window, colors[self.towers_type[edge[0]]], p1, p2, 2)

        pygame.display.update()
