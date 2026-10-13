"""Train a DQN or Double DQN agent on LunarLander-v3.

CLI usage:
    python train.py --episodes 800
    python train.py --episodes 800 --double
"""
import argparse
import json
import os
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch

from dqn_agent import DQNAgent

ENV_ID = "LunarLander-v3"
SOLVED_SCORE = 200.0


def run_name(double: bool) -> str:
    return "double_dqn" if double else "dqn"


def train_agent(
    episodes=800,
    double=False,
    seed=42,
    max_steps=1000,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.995,
    progress_cb=None,
    models_dir="models",
    results_dir="results",
):
    """Train and save the agent. Returns the list of episode rewards."""
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make(ENV_ID)
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        double=double,
    )

    name = run_name(double)
    rewards, window = [], deque(maxlen=100)
    epsilon = eps_start
    solved_at = None

    for ep in range(1, episodes + 1):
        state, _ = env.reset(seed=seed if ep == 1 else None)
        total = 0.0

        for _ in range(max_steps):
            action = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            # Only "terminated" stops bootstrapping; time-limit truncation does not
            agent.step(state, action, reward, next_state, terminated)
            state = next_state
            total += reward
            if terminated or truncated:
                break

        epsilon = max(eps_end, epsilon * eps_decay)
        rewards.append(total)
        window.append(total)
        avg100 = float(np.mean(window))

        if progress_cb is not None:
            progress_cb(ep, episodes, total, avg100, epsilon)
        elif ep % 25 == 0:
            print(f"Episode {ep:4d} | reward {total:8.1f} | avg100 {avg100:8.1f} | eps {epsilon:.3f}")

        if len(window) == 100 and avg100 >= SOLVED_SCORE:
            solved_at = ep
            print(f"Solved in {ep} episodes (avg100 = {avg100:.1f})")
            break

    env.close()
    agent.save(os.path.join(models_dir, f"{name}.pth"))

    with open(os.path.join(results_dir, f"{name}.json"), "w") as f:
        json.dump({"name": name, "double": double, "seed": seed,
                   "solved_at": solved_at, "rewards": rewards}, f)

    return rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--double", action="store_true", help="use Double DQN")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_agent(episodes=args.episodes, double=args.double, seed=args.seed)
