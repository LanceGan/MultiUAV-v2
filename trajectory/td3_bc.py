"""Parameter-shared TD3 with persistent MapGuide demonstration replay."""

import numpy as np
import torch
import torch.nn.functional as F

from trajectory import td3
from trajectory.td3 import SharedTD3


def linear_decay(initial, episode, horizon, floor=0.0):
    """Linear decay to a persistent floor, matching the single-UAV TD3-BC recipe."""
    if floor < 0.0 or floor > initial:
        raise ValueError("floor must be between zero and initial")
    if horizon <= 0:
        return float(floor)
    fraction = max(0.0, 1.0 - float(episode) / float(horizon))
    return float(floor + (initial - floor) * fraction)


def behavior_cloning_loss(predicted, target, demo_mask, max_speed):
    """Masked BC loss on relative heading and speed."""
    count = demo_mask.sum()
    if float(count.item()) == 0.0:
        return predicted.sum() * 0.0
    heading = 1.0 - torch.cos(predicted[:, :1] - target[:, :1])
    speed = ((predicted[:, 1:2] - target[:, 1:2]) / max(float(max_speed), 1e-9)) ** 2
    return ((heading + speed) * demo_mask).sum() / count


class DemoReplayBuffer:
    """Online replay plus a persistent demonstration replay partition."""

    def __init__(self, state_dim, action_dim, max_size=100_000,
                 demo_fraction=0.25):
        if not 0.0 <= demo_fraction <= 1.0:
            raise ValueError("demo_fraction must be in [0, 1]")
        self.max_size = int(max_size)
        self.demo_fraction = float(demo_fraction)
        self.ptr = 0
        self.size = 0
        self.demo_ptr = 0
        self.demo_size = 0

        self.state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((self.max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.done = np.zeros((self.max_size, 1), dtype=np.float32)

        self.demo_state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.demo_action = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.demo_reward = np.zeros((self.max_size, 1), dtype=np.float32)
        self.demo_next_state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.demo_done = np.zeros((self.max_size, 1), dtype=np.float32)

    @property
    def total_size(self):
        return self.size + self.demo_size

    def add(self, state, action, reward, next_state, done, demo=False):
        arrays = (
            ("demo_state", "demo_action", "demo_reward", "demo_next_state", "demo_done")
            if demo else ("state", "action", "reward", "next_state", "done")
        )
        if demo:
            index = self.demo_ptr
            self.demo_ptr = (self.demo_ptr + 1) % self.max_size
            self.demo_size = min(self.demo_size + 1, self.max_size)
        else:
            index = self.ptr
            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)
        getattr(self, arrays[0])[index] = np.asarray(state, dtype=np.float32)
        getattr(self, arrays[1])[index] = np.asarray(action, dtype=np.float32)
        getattr(self, arrays[2])[index] = float(reward)
        getattr(self, arrays[3])[index] = np.asarray(next_state, dtype=np.float32)
        getattr(self, arrays[4])[index] = float(done)

    def __len__(self):
        return self.total_size

    def sample(self, batch_size):
        batch_size = int(batch_size)
        if self.total_size == 0:
            raise ValueError("cannot sample an empty replay buffer")
        if self.demo_size == 0:
            demo_count = 0
        elif self.size == 0:
            demo_count = batch_size
        else:
            demo_count = int(round(batch_size * self.demo_fraction))
        online_count = batch_size - demo_count

        online_idx = (
            np.random.randint(0, self.size, size=online_count)
            if online_count else np.empty(0, dtype=int)
        )
        demo_idx = (
            np.random.randint(0, self.demo_size, size=demo_count)
            if demo_count else np.empty(0, dtype=int)
        )
        state = np.concatenate((self.state[online_idx], self.demo_state[demo_idx]), axis=0)
        action = np.concatenate((self.action[online_idx], self.demo_action[demo_idx]), axis=0)
        reward = np.concatenate((self.reward[online_idx], self.demo_reward[demo_idx]), axis=0)
        next_state = np.concatenate(
            (self.next_state[online_idx], self.demo_next_state[demo_idx]), axis=0
        )
        done = np.concatenate((self.done[online_idx], self.demo_done[demo_idx]), axis=0)
        demo_mask = np.concatenate((
            np.zeros((online_count, 1), dtype=np.float32),
            np.ones((demo_count, 1), dtype=np.float32),
        ), axis=0)

        permutation = np.random.permutation(batch_size)
        arrays = (state, action, reward, next_state, done, demo_mask)
        arrays = tuple(array[permutation] for array in arrays)
        return tuple(
            torch.tensor(array, dtype=torch.float32, device=td3.device)
            for array in arrays
        )


class SharedTD3BC(SharedTD3):
    """Parameter-shared TD3 whose actor update includes masked BC regularization."""

    def __init__(self, *args, bc_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.bc_weight = float(bc_weight)
        self.last_bc_loss = 0.0

    def train(self, replay_buffer):
        state, action, reward, next_state, done, demo_mask = replay_buffer.sample(
            self.batch_size
        )

        with torch.no_grad():
            noise = torch.clamp(
                torch.randn_like(action) * self.policy_noise,
                -self.noise_clip,
                self.noise_clip,
            )
            next_action = self.actor_target(next_state) + noise
            next_action = torch.max(
                torch.min(next_action, self.max_action),
                self.min_action,
            )
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1.0 - done) * self.gamma * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = (
            F.mse_loss(current_q1, target_q)
            + F.mse_loss(current_q2, target_q)
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        self._update_counter += 1
        actor_loss = None
        bc_loss = 0.0
        if self._update_counter % self.policy_update_delay == 0:
            predicted = self.actor(state)
            q_term = -self.critic.Q1(state, predicted).mean()
            bc_term = behavior_cloning_loss(
                predicted, action, demo_mask,
                self.max_action[0].detach().cpu().item(),
            )
            actor_loss = q_term + self.bc_weight * bc_term
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            bc_loss = float(bc_term.detach().cpu().item())
            self.last_bc_loss = bc_loss

            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1.0 - self.tau) * target_param.data
                )
            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1.0 - self.tau) * target_param.data
                )

        return {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "actor_loss": (
                float(actor_loss.detach().cpu().item())
                if actor_loss is not None else None
            ),
            "bc_loss": float(bc_loss),
        }
