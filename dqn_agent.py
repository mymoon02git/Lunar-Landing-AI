"""DQN / Double DQN agent for LunarLander."""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    """MLP that maps a state (8 values) to Q-values for each action (4)."""

    def __init__(self, state_dim=8, action_dim=4, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Fixed-size buffer of past transitions, sampled uniformly at random."""

    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.tensor(np.array(s), dtype=torch.float32, device=DEVICE),
            torch.tensor(a, dtype=torch.int64, device=DEVICE).unsqueeze(1),
            torch.tensor(r, dtype=torch.float32, device=DEVICE).unsqueeze(1),
            torch.tensor(np.array(s2), dtype=torch.float32, device=DEVICE),
            torch.tensor(d, dtype=torch.float32, device=DEVICE).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        state_dim=8,
        action_dim=4,
        lr=5e-4,
        gamma=0.99,
        batch_size=64,
        buffer_size=100_000,
        tau=1e-3,
        update_every=4,
        double=False,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.update_every = update_every
        self.double = double
        self.t_step = 0

        self.q_local = QNetwork(state_dim, action_dim).to(DEVICE)
        self.q_target = QNetwork(state_dim, action_dim).to(DEVICE)
        self.q_target.load_state_dict(self.q_local.state_dict())
        self.optimizer = optim.Adam(self.q_local.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)

    def act(self, state, epsilon=0.0):
        """Epsilon-greedy action selection."""
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        self.q_local.eval()
        with torch.no_grad():
            q_values = self.q_local(s)
        self.q_local.train()
        return int(q_values.argmax(dim=1).item())

    def step(self, state, action, reward, next_state, done):
        """Store a transition and learn every `update_every` steps."""
        self.memory.push(state, action, reward, next_state, float(done))
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) >= self.batch_size:
            self._learn(self.memory.sample(self.batch_size))

    def _learn(self, batch):
        s, a, r, s2, d = batch
        q_pred = self.q_local(s).gather(1, a)

        with torch.no_grad():
            if self.double:
                # Double DQN: local net picks the action, target net evaluates it
                best_a = self.q_local(s2).argmax(dim=1, keepdim=True)
                next_q = self.q_target(s2).gather(1, best_a)
            else:
                next_q = self.q_target(s2).max(dim=1, keepdim=True)[0]
            q_target = r + self.gamma * next_q * (1.0 - d)

        loss = F.smooth_l1_loss(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_local.parameters(), 10.0)
        self.optimizer.step()

        # Soft update of the target network
        for tp, lp in zip(self.q_target.parameters(), self.q_local.parameters()):
            tp.data.copy_(self.tau * lp.data + (1.0 - self.tau) * tp.data)

    def save(self, path):
        torch.save(self.q_local.state_dict(), path)

    def load(self, path):
        self.q_local.load_state_dict(torch.load(path, map_location=DEVICE))
        self.q_target.load_state_dict(self.q_local.state_dict())
        self.q_local.eval()
