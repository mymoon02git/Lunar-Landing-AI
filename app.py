
import glob
import json
import os
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from dqn_agent import DQNAgent
from train import ENV_ID, run_name, train_agent

MODELS_DIR, RESULTS_DIR = "models", "results"


def plot_curves(curves, ylabel="Reward"):
    """Matplotlib chart (avoids pyarrow, which st.line_chart needs)."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for label, series in curves.items():
        ax.plot(list(series), label=label)
    ax.axhline(200, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    return fig


def show_plot(container, curves, ylabel="Reward"):
    fig = plot_curves(curves, ylabel)
    container.pyplot(fig)
    plt.close(fig)


def show_table(rows):
    """Markdown table (avoids pyarrow, which st.dataframe needs)."""
    cols = list(rows[0].keys())
    md = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    md += "\n".join("| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows)
    st.markdown(md)

st.set_page_config(page_title="Lunar Escape AI", layout="wide")
st.title(" Lunar Escape AI")
st.caption("Deep reinforcement learning (DQN & Double DQN) landing a spacecraft on the Moon")

tab_about, tab_train, tab_watch, tab_results = st.tabs(
    ["About", "Train", "Watch agent", "Results"]
)


# ---------------------------------------------------------------- About
with tab_about:
    st.subheader("The problem")
    st.write(
        "The lander must touch down softly between the two flags. Fuel is unlimited, "
        "but crashing, drifting away or firing engines wastes reward."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """**State (8 values)**
- x, y position
- x, y velocity
- angle, angular velocity
- left leg contact, right leg contact"""
        )
    with c2:
        st.markdown(
            """**Actions (4)**
- 0: do nothing
- 1: fire left engine
- 2: fire main engine
- 3: fire right engine"""
        )
    st.markdown(
        """**Method**
- Q-network (MLP 8 → 128 → 128 → 4) approximates Q(s, a)
- Experience replay breaks correlation between consecutive samples
- Soft-updated target network stabilises training
- Epsilon-greedy exploration with decay
- Optional **Double DQN** reduces Q-value overestimation

The environment counts as solved at an average reward of **200** over 100 episodes."""
    )


# ---------------------------------------------------------------- Train
with tab_train:
    st.subheader("Train an agent")
    col1, col2, col3 = st.columns(3)
    episodes = col1.slider("Max episodes", 100, 2000, 600, step=50)
    double = col2.checkbox("Use Double DQN", value=False)
    seed = col3.number_input("Seed", value=42, step=1)
    st.info("Training stops early once the 100-episode average reaches 200. "
            "Expect several minutes on CPU.")

    if st.button("Start training", type="primary"):
        progress = st.progress(0.0)
        status = st.empty()
        chart = st.empty()
        history = []

        def on_progress(ep, total_eps, reward, avg100, eps):
            history.append({"reward": reward, "avg100": avg100})
            progress.progress(ep / total_eps)
            status.write(f"Episode {ep}/{total_eps} | reward {reward:.1f} | "
                         f"avg100 {avg100:.1f} | epsilon {eps:.3f}")
            if ep % 10 == 0:
                show_plot(chart, {"Reward": [h["reward"] for h in history],
                                  "Average (100)": [h["avg100"] for h in history]})

        rewards = train_agent(episodes=episodes, double=double, seed=int(seed),
                              progress_cb=on_progress)
        show_plot(chart, {"Reward": [h["reward"] for h in history],
                          "Average (100)": [h["avg100"] for h in history]})
        progress.progress(1.0)
        st.success(f"Finished after {len(rewards)} episodes. "
                   f"Saved models/{run_name(double)}.pth")


# ---------------------------------------------------------------- Watch
def rollout(agent_choice, seed, delay, placeholder):
    env = gym.make(ENV_ID, render_mode="rgb_array")
    agent = None
    if agent_choice != "Random agent":
        agent = DQNAgent(double=(agent_choice == "Double DQN"))
        agent.load(os.path.join(MODELS_DIR, f"{run_name(agent_choice == 'Double DQN')}.pth"))

    state, _ = env.reset(seed=int(seed))
    total, steps, terminated, truncated = 0.0, 0, False, False
    while not (terminated or truncated):
        action = env.action_space.sample() if agent is None else agent.act(state, 0.0)
        state, reward, terminated, truncated, _ = env.step(action)
        total += reward
        steps += 1
        if steps % 2 == 0:  # skip frames to keep the UI fast
            placeholder.image(env.render(), channels="RGB", use_container_width=True)
            time.sleep(delay)
    env.close()
    return total, steps, state


with tab_watch:
    st.subheader("Watch an agent play")
    options = ["Random agent"]
    for label, dbl in [("DQN", False), ("Double DQN", True)]:
        if os.path.exists(os.path.join(MODELS_DIR, f"{run_name(dbl)}.pth")):
            options.append(label)

    c1, c2, c3 = st.columns(3)
    choice = c1.selectbox("Agent", options)
    play_seed = c2.number_input("Episode seed", value=7, step=1)
    delay = c3.slider("Frame delay (s)", 0.0, 0.1, 0.01)
    if len(options) == 1:
        st.warning("No trained model found yet. Train one in the Train tab first.")

    if st.button("Run episode"):
        frame_box = st.empty()
        total, steps, final_state = rollout(choice, play_seed, delay, frame_box)
        landed = bool(final_state[6] and final_state[7])  # both legs touching
        m1, m2, m3 = st.columns(3)
        m1.metric("Total reward", f"{total:.1f}")
        m2.metric("Steps", steps)
        m3.metric("Outcome", "Landed " if landed and total > 100 else "Crashed / drifted ")


# ---------------------------------------------------------------- Results
with tab_results:
    st.subheader("Training comparison")
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json")))
    if not files:
        st.info("No results yet. Train at least one agent first.")
    else:
        window = st.slider("Moving average window", 5, 100, 50)
        curves, rows = {}, []
        for path in files:
            with open(path) as f:
                data = json.load(f)
            r = pd.Series(data["rewards"])
            label = "Double DQN" if data["double"] else "DQN"
            curves[label] = r.rolling(window, min_periods=1).mean()
            rows.append({
                "Agent": label,
                "Episodes trained": len(r),
                "Solved at episode": data["solved_at"] or "not solved",
                "Best episode": round(r.max(), 1),
                "Last-100 average": round(r.tail(100).mean(), 1),
            })
        show_plot(st, curves, ylabel=f"Reward (moving avg, window {window})")
        show_table(rows)
