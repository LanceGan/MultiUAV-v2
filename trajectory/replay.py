"""Experience replay buffer for the parameter-shared independent TD3 (Task 7).

A single buffer pools every UAV's individual transition ``(s, a, r, s', done)``;
the shared networks train on the pooled samples (parameter sharing). ``done`` is
the *episode-level* terminal flag (``info["terminal"]``, true completion only —
not truncation), not a per-UAV flag, so target Q bootstrap correctly at the end
of a completed episode (and truncated episodes bootstrap with ``done=0``).
"""

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(2e5)):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((self.max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.done = np.zeros((self.max_size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        """Store one transition. ``done`` is float (1.0 = episode terminal)."""
        self.state[self.ptr] = np.asarray(state, dtype=np.float32)
        self.action[self.ptr] = np.asarray(action, dtype=np.float32)
        self.reward[self.ptr] = float(reward)
        self.next_state[self.ptr] = np.asarray(next_state, dtype=np.float32)
        self.done[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(device),
            torch.FloatTensor(self.action[ind]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device),
            torch.FloatTensor(self.next_state[ind]).to(device),
            torch.FloatTensor(self.done[ind]).to(device),
        )

    def __len__(self):
        return self.size

    def is_ready(self, batch_size):
        return self.size >= batch_size
