import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch
import gymnasium as gym
from torch.distributions import Categorical
from collections import deque


class Policy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.expand = nn.Linear(input_size, hidden_size)
        self.compress = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # logits are before Softmax layer. Also don't need activation after compress layer because softmax is the non-linearity
        return F.softmax(self.compress(F.relu(self.expand(x))), dim=-1)

env = gym.make("CartPole-v1")
eval_env = gym.make("CartPole-v1")
obs_size = env.observation_space.shape[0]
act_size = env.action_space.n

cartpole_hyperparameters = {
    "h_size": 32,
    "n_training_episodes": 1000,
    "n_evaluation_episodes": 30,
    "max_t": 100,
    "gamma": 0.99,
    "lr": 1e-2,
    "env_id": "CartPole-v1",
    "state_space": obs_size,
    "action_space": act_size,
}

policy = Policy(cartpole_hyperparameters["state_space"], cartpole_hyperparameters["h_size"], cartpole_hyperparameters["action_space"])
optimizer = Adam(policy.parameters(), lr=cartpole_hyperparameters["lr"])


def train_reward_to_go(policy, optimizer, num_eps, max_steps):
    returns = []
    for eps in range(1, num_eps+1):
        eps_rewards = []
        logprobs = []
        obs, _ = env.reset()
    
        for step in range(max_steps):

            action_probs = policy(torch.from_numpy(obs))
            action = torch.multinomial(action_probs, 1).item()
            logp = torch.log(action_probs)[action]
            
            obs, rwd, termination, truncation, _ = env.step(action)
            
            eps_rewards.append(rwd)
            logprobs.append(logp.unsqueeze(0))  # make dim=1 instead of dim=0
            if termination or truncation:
                break
    
        ## 1-sample estimate of J(theta) = Sum_t {log pi(a_t|s_t) * R(tau)}
        
        # reward-to-go: prefix-sum iterating backwards
        eps_return = deque(maxlen=max_steps)
        eps_return.append(eps_rewards[-1])
        for t in range(len(eps_rewards)-1)[::-1]:
            eps_return.appendleft(eps_rewards[t] + eps_return[0])

        returns.append(eps_return[0])

        loss = -(torch.cat(logprobs) * torch.tensor(eps_return)).sum()  # dot product
        if eps % (num_eps//10) == 0:
            print(f"episode: {eps}, avg return: {np.mean(returns)}")
            returns = []
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

