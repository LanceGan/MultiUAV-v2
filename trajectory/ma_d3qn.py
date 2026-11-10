"""Parameter-shared multi-agent D3QN with DQfD-style demonstrations."""

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class DiscreteActionCodec:
    """Hover plus 16 relative headings at two speed levels."""

    HEADING_COUNT = 16

    def __init__(self, v_max):
        self.v_max = float(v_max)
        self.speeds = (0.5 * self.v_max, self.v_max)
        headings = [
            _wrap_angle(2.0 * math.pi * index / self.HEADING_COUNT)
            for index in range(self.HEADING_COUNT)
        ]
        self.actions = np.asarray(
            [[0.0, 0.0]]
            + [[speed, heading] for speed in self.speeds for heading in headings],
            dtype=np.float32,
        )

    def __len__(self):
        return len(self.actions)

    def decode(self, action_index):
        return self.actions[int(action_index)].copy()

    def encode(self, action):
        speed, heading = float(action[0]), float(action[1])
        if speed < 0.25 * self.v_max:
            return 0
        speed_index = int(np.argmin(np.abs(np.asarray(self.speeds) - speed)))
        step = 2.0 * math.pi / self.HEADING_COUNT
        heading_index = int(round(_wrap_angle(heading) / step)) % self.HEADING_COUNT
        return 1 + speed_index * self.HEADING_COUNT + heading_index


class NStepAssembler:
    """Accumulate per-decision transitions into fixed-horizon n-step records."""

    def __init__(self, n, gamma):
        self.n = int(n)
        self.gamma = float(gamma)
        if self.n < 1:
            raise ValueError("n must be positive")
        self.pending = []

    def _record(self, horizon):
        state, action, _reward, _next_state, _done = self.pending[0]
        nstep_reward = sum(
            (self.gamma ** index) * self.pending[index][2]
            for index in range(horizon)
        )
        _, _, _, nstep_next, nstep_done = self.pending[horizon - 1]
        return (
            state, action, self.pending[0][2], self.pending[0][3],
            self.pending[0][4], nstep_reward, nstep_next,
            nstep_done, self.gamma ** horizon,
        )

    def push(self, state, action, reward, next_state, done):
        self.pending.append((state, action, reward, next_state, done))
        records = []
        if done:
            while self.pending:
                records.append(self._record(len(self.pending)))
                self.pending.pop(0)
        elif len(self.pending) >= self.n:
            records.append(self._record(self.n))
            self.pending.pop(0)
        return records


class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        features = self.feature(state)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class DiscreteReplayBuffer:
    """Online transitions plus a persistent demonstration partition."""

    def __init__(self, state_dim, capacity=200_000, demo_fraction=0.25):
        self.capacity = int(capacity)
        self.demo_fraction = float(demo_fraction)
        if not 0.0 <= self.demo_fraction <= 1.0:
            raise ValueError("demo_fraction must be in [0, 1]")
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.nstep_rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.nstep_next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.nstep_dones = np.zeros((capacity, 1), dtype=np.float32)
        self.nstep_gammas = np.zeros((capacity, 1), dtype=np.float32)

        self.demo_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.demo_actions = np.zeros((capacity, 1), dtype=np.int64)
        self.demo_rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.demo_next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.demo_dones = np.zeros((capacity, 1), dtype=np.float32)
        self.demo_nstep_rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.demo_nstep_next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.demo_nstep_dones = np.zeros((capacity, 1), dtype=np.float32)
        self.demo_nstep_gammas = np.zeros((capacity, 1), dtype=np.float32)
        self.position = 0
        self.size = 0
        self.demo_position = 0
        self.demo_size = 0

    def __len__(self):
        return self.size + self.demo_size

    def add(self, record, demo=False):
        arrays = (
            (
                self.demo_states, self.demo_actions, self.demo_rewards,
                self.demo_next_states, self.demo_dones,
                self.demo_nstep_rewards, self.demo_nstep_next_states,
                self.demo_nstep_dones, self.demo_nstep_gammas,
            )
            if demo else (
                self.states, self.actions, self.rewards, self.next_states,
                self.dones, self.nstep_rewards, self.nstep_next_states,
                self.nstep_dones, self.nstep_gammas,
            )
        )
        if demo:
            index = self.demo_position
            self.demo_position = (self.demo_position + 1) % self.capacity
            self.demo_size = min(self.demo_size + 1, self.capacity)
        else:
            index = self.position
            self.position = (self.position + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
        state, action, reward, next_state, done, nreward, nnext, ndone, ngamma = record
        arrays[0][index] = state
        arrays[1][index, 0] = int(action)
        arrays[2][index, 0] = float(reward)
        arrays[3][index] = next_state
        arrays[4][index, 0] = float(done)
        arrays[5][index, 0] = float(nreward)
        arrays[6][index] = nnext
        arrays[7][index, 0] = float(ndone)
        arrays[8][index, 0] = float(ngamma)

    def sample(self, batch_size):
        if not len(self):
            raise ValueError("cannot sample an empty buffer")
        if self.demo_size == 0:
            demo_count = 0
        elif self.size == 0:
            demo_count = int(batch_size)
        else:
            demo_count = int(round(batch_size * self.demo_fraction))
        online_count = int(batch_size) - demo_count
        online = (
            np.random.randint(0, self.size, size=online_count)
            if online_count else np.empty(0, dtype=int)
        )
        demo = (
            np.random.randint(0, self.demo_size, size=demo_count)
            if demo_count else np.empty(0, dtype=int)
        )
        arrays = [
            np.concatenate((online_array[online], demo_array[demo]), axis=0)
            for online_array, demo_array in zip(
                (self.states, self.actions, self.rewards, self.next_states,
                 self.dones, self.nstep_rewards, self.nstep_next_states,
                 self.nstep_dones, self.nstep_gammas),
                (self.demo_states, self.demo_actions, self.demo_rewards,
                 self.demo_next_states, self.demo_dones, self.demo_nstep_rewards,
                 self.demo_nstep_next_states, self.demo_nstep_dones,
                 self.demo_nstep_gammas),
            )
        ]
        demo_mask = np.concatenate((
            np.zeros((online_count, 1), dtype=np.float32),
            np.ones((demo_count, 1), dtype=np.float32),
        ))
        permutation = np.random.permutation(int(batch_size))
        arrays = [array[permutation] for array in arrays] + [demo_mask[permutation]]
        return tuple(
            torch.tensor(array, dtype=torch.float32, device=device)
            if array.dtype != np.int64 else
            torch.tensor(array, dtype=torch.long, device=device)
            for array in arrays
        )


def _double_dqn_targets(rewards, dones, next_states, online, target, gamma):
    with torch.no_grad():
        next_actions = online(next_states).argmax(dim=1, keepdim=True)
        next_values = target(next_states).gather(1, next_actions)
    return rewards + gamma * (1.0 - dones) * next_values


def _margin_loss(q_values, actions, demo_mask, margin):
    count = demo_mask.sum()
    if float(count.item()) == 0.0:
        return q_values.sum() * 0.0
    penalty = torch.full_like(q_values, float(margin))
    penalty.scatter_(1, actions, 0.0)
    augmented_max = (q_values + penalty).max(dim=1, keepdim=True).values
    shortfall = augmented_max - q_values.gather(1, actions)
    return (shortfall * demo_mask).sum() / count


class MAD3QNAgent:
    def __init__(
        self, state_dim, action_dim, hidden_dim=256, learning_rate=1e-4,
        gamma=0.995, batch_size=256, target_update_interval=1000,
        gradient_clip=10.0, margin=0.8,
    ):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.target_update_interval = int(target_update_interval)
        self.gradient_clip = float(gradient_clip)
        self.margin = float(margin)
        self.bc_weight = 0.0
        self.n_step_weight = 1.0
        self.online = DuelingQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target = DuelingQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target.load_state_dict(self.online.state_dict())
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.update_count = 0

    def select_action(self, state, epsilon=0.0):
        if np.random.rand() < epsilon:
            return int(np.random.randint(self.action_dim))
        with torch.no_grad():
            state_tensor = torch.tensor(
                np.asarray(state, dtype=np.float32), dtype=torch.float32,
                device=device,
            ).reshape(1, -1)
            return int(self.online(state_tensor).argmax(dim=1).item())

    def train(self, replay_buffer):
        (states, actions, rewards, next_states, dones, nstep_rewards,
         nstep_next_states, nstep_dones, nstep_gammas, demo_mask) = (
            replay_buffer.sample(self.batch_size)
        )
        q_values = self.online(states)
        predictions = q_values.gather(1, actions)
        targets = _double_dqn_targets(
            rewards, dones, next_states, self.online, self.target, self.gamma
        )
        td_loss = F.smooth_l1_loss(predictions, targets)
        nstep_targets = _double_dqn_targets(
            nstep_rewards, nstep_dones, nstep_next_states,
            self.online, self.target, nstep_gammas,
        )
        nstep_loss = F.smooth_l1_loss(predictions, nstep_targets)
        margin_loss = _margin_loss(q_values, actions, demo_mask, self.margin)
        loss = td_loss + self.n_step_weight * nstep_loss + self.bc_weight * margin_loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.gradient_clip)
        self.optimizer.step()
        self.update_count += 1
        if self.update_count % self.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return {
            "loss": float(loss.detach().cpu().item()),
            "td_loss": float(td_loss.detach().cpu().item()),
            "nstep_loss": float(nstep_loss.detach().cpu().item()),
            "margin_loss": float(margin_loss.detach().cpu().item()),
        }

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": "ma-d3qn-v1",
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_count": self.update_count,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=device)
        if (
            checkpoint.get("schema") != "ma-d3qn-v1"
            or int(checkpoint.get("state_dim", -1)) != self.state_dim
            or int(checkpoint.get("action_dim", -1)) != self.action_dim
        ):
            raise ValueError("incompatible MA-D3QN checkpoint")
        self.online.load_state_dict(checkpoint["online"])
        self.target.load_state_dict(checkpoint["target"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.update_count = int(checkpoint.get("update_count", 0))
