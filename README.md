# 🚀 Lunar Escape AI – Deep Q-Learning on LunarLander

A deep reinforcement learning project where an agent learns to land a spacecraft using **DQN** and **Double DQN** (PyTorch + Gymnasium), with a Streamlit app to train, watch and compare agents.

## Why deep RL here
Earlier projects (GridWorld, Maze Escape, Cliff Walking) used tabular Q-learning/SARSA. LunarLander has a continuous 8-dimensional state, so a Q-table is impossible and a neural network approximates Q(s, a).

## Environment
- **State (8):** x, y, vx, vy, angle, angular velocity, left-leg contact, right-leg contact
- **Actions (4):** nothing, left engine, main engine, right engine
- **Solved:** average reward ≥ 200 over 100 episodes

## Method
- MLP Q-network (8 → 128 → 128 → 4)
- Experience replay (100k transitions), batch size 64
- Soft-updated target network (tau = 1e-3)
- Epsilon-greedy exploration (1.0 → 0.01, decay 0.995 per episode)
- Optional Double DQN target
- Adam, lr = 5e-4, gamma = 0.99, Huber loss, gradient clipping

## Project structure
```
lunar_escape_ai/
├── app.py            # Streamlit app (train / watch / compare)
├── dqn_agent.py      # Q-network, replay buffer, DQN / Double DQN agent
├── train.py          # Training loop + CLI
├── requirements.txt
├── models/           # saved weights (created on training)
└── results/          # reward logs (created on training)
```

## Setup
```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install swig
pip install -r requirements.txt
```

## Run
```bash
# train from the command line
python train.py --episodes 800
python train.py --episodes 800 --double

# or use the app
streamlit run app.py
```

## Results
| Agent | Episodes to solve | Last-100 avg |
|-------|-------------------|--------------|
| Random | – | ~ -180 |
| DQN | _fill in_ | _fill in_ |
| Double DQN | _fill in_ | _fill in_ |

Add your reward curve screenshot and a short GIF of the trained lander here.

## What I learned
- Why replay and target networks stabilise training
- Treating time-limit truncation differently from true termination
- Overestimation bias in Q-learning and how Double DQN reduces it
