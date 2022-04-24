import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from torch.distributions import Categorical
from itertools import count
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic_Batch():

    def __init__(self, env, actor_lr, critic_lr, gamma):
        self.env = env
        self.actor = Actor(env)
        self.critic = Critic(env)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.optim_actor = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.gamma = gamma

    def update(self, values, rewards, log_probs):
        ####### YOUR CODE IN HERE #######
        w_loss = 0
        for i in range(len(rewards)):
            sample_loss = rewards[i]
            for j in range(i + 1, len(rewards)):
                sample_loss += (self.gamma ** (j - i)) * rewards[j]
            sample_loss = sample_loss - values[i]
            w_loss += sample_loss

        self.optim_critic.zero_grad()
        w_loss.backward()
        self.optim_critic.step()

        theta_loss = 0
        for i in range(len(rewards)):
            sample_loss = rewards[i]
            for j in range(i + 1, len(rewards)):
                sample_loss += (self.gamma ** (j - i)) * rewards[j]
            sample_loss = (-1) * log_probs[i] * (sample_loss - values[i]).detach()
            theta_loss += sample_loss

        self.optim_actor.zero_grad()
        theta_loss.backward()
        self.optim_actor.step()

        #################################


    def learn_task(self, n_episodes):

        episode_rewards = []
        avg_rewards = []

        for episode in tqdm(range(n_episodes)):
            state = self.env.reset()
            rewards = []
            values = []
            log_probs = []
            state = torch.FloatTensor(state).to(device)

            for i in count():

                dist, value = self.actor(state), self.critic(state)
                action = dist.sample()  # We sample an action index from our softmax (multinomial) distribution.
                next_state, reward, done, _ = self.env.step(int(action.cpu().numpy()))  # gym does not like Pytorch Tensors
                if i % 10 == 0:
                    self.env.render()
                rewards.append(reward)

                next_state = torch.FloatTensor(next_state).to(device)
                next_value = self.critic(next_state)
                log_prob = dist.log_prob(action).unsqueeze(0)

                # I'll get you started...
                log_probs.append(log_prob)
                values.append(value)
                # rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
                # masks.append(torch.tensor([1-done], dtype=torch.float, device=device))

                if done:
                    # Anything you want printed or logged at the end of an episode...
                    ep_return = np.sum(rewards)
                    episode_rewards.append(ep_return)
                    break

                state = next_state

            ####### YOUR CODE IN HERE #######

            # (this is the part with the delta, the loss functions...)

            #################################

            self.update(values, rewards, log_probs)  # YOUR CODE HERE

            if episode >= 10:
                avg_rewards.append(np.mean(episode_rewards[-10:]))
                if avg_rewards[-1] >= 3000:
                    print("Solved on episode ", episode)
                    return episode_rewards, avg_rewards

        return episode_rewards, avg_rewards

    def play(self):
        state = self.env.reset()
        state = torch.FloatTensor(state).to(device)

        for i in count():

            self.env.clock.tick(30)
            dist, value = self.actor(state), self.critic(state)
            action = dist.sample()  # We sample an action index from our softmax (multinomial) distribution.
            next_state, reward, done, _ = self.env.step(int(action.cpu().numpy()))  # gym does not like Pytorch Tensors
            self.env.render()

            next_state = torch.FloatTensor(next_state).to(device)

            if done:
                # Anything you want printed or logged at the end of an episode...
                break

            state = next_state


class Actor(nn.Module):
    """
    Feel free to change the architecture for different tasks!
    """
    def __init__(self, env):
        super(Actor, self).__init__()
        self.state_size = env.observation_space.shape[0]
        if type(env.action_space) == gym.spaces.box.Box:
            self.action_size = env.action_space.shape[0]
        else:
            self.action_size = env.action_space.n
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        # Note the conversion to Pytorch distribution.
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution


class Critic(nn.Module):
    """
    Feel free to change the architecture for different tasks!
    """
    def __init__(self, env):
        super(Critic, self).__init__()
        self.state_size = env.observation_space.shape[0]
        if type(env.action_space) == gym.spaces.box.Box:
            self.action_size = env.action_space.shape[0]
        else:
            self.action_size = env.action_space.n
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1) # Note the single value - this is V(s)!

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value