import json
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from base64 import b64encode


def set_seed(env, seed):
    """Helper function to set the seeds when needed"""
    env.seed(seed)  # Environment seed
    env.action_space.seed(seed)  # Seed for env.action_space.sample()
    np.random.seed(seed)  # Numpy seed
    torch.manual_seed(seed)  # PyTorch seed
    random.seed(seed)  # seed for Python random library


class MultiAgentTrainer:
    """A class to train agents in a multi-agent environment"""

    def __init__(self, env, agents, reset_agents, seed=None):
        # Save parameters
        self.env, self.agents, self.seed = env, agents, seed

        # Create log of rewards and reset agents
        self.rewards_log = {key: [] for key in self.agents.keys()}
        self.reset(reset_agents)

    def reset(self, reset_agents):
        # Set seed for reproducibility
        if self.seed is not None:
            set_seed(self.env, self.seed)

        # Reset agents and clear log of rewards
        for key, agent in self.agents.items():
            self.rewards_log[key].clear()

            if reset_agents:
                agent.reset()

    def train(self, n_steps):
        # Reset env. and get initial observations
        states = self.env.reset()

        # Set greedy flag
        for key, agent in self.agents.items():
            agent.is_greedy = False

        for i in tqdm(range(n_steps), 'Training agents'):
            # Select actions based on current states
            actions = {key: agent.act(states[key]) for key, agent in self.agents.items()}

            # Perform the selected action
            next_states, rewards, dones, _ = self.env.step(actions)

            # Learn from experience
            for key, agent in self.agents.items():
                agent.learn(states[key], actions[key], rewards[key], next_states[key], dones[key])
                self.rewards_log[key].append(rewards[key])
            states = next_states


def test_agents(env, agents, n_steps, seed=None):
    """Function to test agents"""

    # Initialization
    if seed is not None:
        set_seed(env, seed=seed)
    states = env.reset()
    rewards_log = defaultdict(list)

    # Set greedy flag
    for key, agent in agents.items():
        agent.is_greedy = True

    for _ in tqdm(range(n_steps), 'Testing agents'):
        # Select actions based on current states
        with torch.no_grad():
            actions = {key: agent.act(states[key]) for key, agent in agents.items()}

        # Perform the selected action
        next_states, rewards, dones, _ = env.step(actions)

        # Save rewards
        for key, reward in rewards.items():
            rewards_log[key].append(reward)

        states = next_states

    return rewards_log


def plot_cumulative_rewards(rewards_log, events={'delivery': [1], 'crash': [-1]}, drones_labels=None, ax=None):
    # Creat figure etc.. if ax none
    create_figure = (ax is None)
    if create_figure:
        fig = plt.figure(figsize=(12, 4))
        ax = fig.gca()

    # Plot rewards
    for key, rewards in rewards_log.items():
        # Drone name
        if (drones_labels is None) or (key not in drones_labels):
            drone_name = 'Drone {}'.format(key)
        else:
            drone_name = drones_labels[key]

        # Reward stats
        label = '{} - reward: {:.3f}±{:.3f}'.format(drone_name, np.mean(rewards), np.std(rewards))

        # Events stats
        for event, rewards_values in events.items():
            event_mask = np.isin(rewards, rewards_values)
            label += ' ' + '{}: {:.1f}% ({})'.format(event, 100 * np.mean(event_mask), np.sum(event_mask))

        # Plot cumulative sum with stats
        cumsum = np.cumsum(rewards)
        idxs = range(1, len(cumsum) + 1)
        ax.step(idxs, cumsum, label=label)

    ax.set_xlabel('Step')
    ax.set_ylabel('Cumulative reward')
    ax.legend()

    if create_figure:
        plt.show()


def plot_rolling_rewards(rewards_log, window=None, hline=None, events={'delivery': [1], 'crash': [-1]}, drones_labels=None, ax=None):
    # Creat figure etc.. if ax none
    create_figure = (ax is None)
    if create_figure:
        fig = plt.figure(figsize=(12, 4))
        ax = fig.gca()

    for key, rewards in rewards_log.items():
        # Drone name
        if (drones_labels is None) or (key not in drones_labels):
            drone_name = 'Drone {}'.format(key)
        else:
            drone_name = drones_labels[key]

        # Events stats
        label = '{}'.format(drone_name) + ' -' if len(events) > 0 else ''
        for event, rewards_values in events.items():
            event_mask = np.isin(rewards, rewards_values)
            label += ' ' + '{}: {:.1f}% ({})'.format(event, 100 * np.mean(event_mask), np.sum(event_mask))

        # Set default for window size
        window = int(len(rewards) / 10) if window is None else window

        # Plot rolling mean
        rolling_mean = pd.Series(rewards).rolling(window).mean()
        steps = range(1, len(rewards) + 1)
        ax.plot(steps, rolling_mean, label=label)

    if hline is not None:
        ax.axhline(hline, label='target value', c='C0', linestyle='--')

    # Add title, labels and legend
    ax.set_xlabel('Steps (rolling window: {})'.format(window))
    ax.set_ylabel('Rewards')
    ax.legend()

    if create_figure:
        plt.show()


def collect_frames(env, agents, n_steps=60, seed=None):
    """Run a greedy episode and return the list of rgb_array frames."""
    # Initialization
    if seed is not None:
        set_seed(env, seed=seed)
    states = env.reset()

    # Set greedy flag
    for key, agent in agents.items():
        agent.is_greedy = True

    # Run agents
    frames = []
    for _ in tqdm(range(n_steps), 'Running agents', unit='frame'):
        # Select actions based on current states
        actions = {key: agent.act(states[key]) for key, agent in agents.items()}

        # Perform the selected action
        next_states, rewards, dones, _ = env.step(actions)
        states = next_states

        # Save frame
        frames.append(env.render(mode='rgb_array'))

    return frames


def render_video(env, agents, video_path, n_steps=60, fps=1, seed=None):
    import os
    try:
        # moviepy >= 2.0
        from moviepy import ImageClip, concatenate_videoclips
    except ImportError:
        # moviepy 1.x (e.g. the version preinstalled on Colab)
        from moviepy.editor import ImageClip, concatenate_videoclips

    frames = collect_frames(env, agents, n_steps=n_steps, seed=seed)

    # Create the parent directory if needed
    directory = os.path.dirname(video_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # set_duration was renamed to with_duration in moviepy 2.0
    def with_duration(clip, duration):
        setter = getattr(clip, 'with_duration', None) or clip.set_duration
        return setter(duration)

    # Create video
    clips = [with_duration(ImageClip(frame), fps) for frame in frames]
    concat_clip = concatenate_videoclips(clips, method="compose")
    concat_clip.write_videofile(video_path, fps=24)


def render_html(env, agents, html_path, n_steps=60, fps=1, seed=None):
    """Export a greedy episode as a self-contained, browser-playable HTML animation.

    Unlike render_video this needs no ffmpeg: each frame is embedded as a base64
    PNG and played back with a small built-in JavaScript player (play/pause,
    scrubber, speed). ``fps`` is the playback speed in frames per second.
    Returns the path written.
    """
    import io
    import os
    from PIL import Image

    frames = collect_frames(env, agents, n_steps=n_steps, seed=seed)

    # Encode each frame as a base64 PNG data URI
    data_uris = []
    for frame in frames:
        buffer = io.BytesIO()
        Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(buffer, format='PNG')
        data_uris.append('data:image/png;base64,' + b64encode(buffer.getvalue()).decode())

    interval_ms = int(round(1000 / fps)) if fps else 200
    html = (_HTML_TEMPLATE
            .replace('__FRAMES__', json.dumps(data_uris))
            .replace('__INTERVAL__', str(interval_ms))
            .replace('__LAST__', str(max(len(data_uris) - 1, 0))))

    # Create the parent directory if needed, then write the file
    directory = os.path.dirname(html_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(html_path, 'w') as f:
        f.write(html)

    return html_path


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DroneRL episode</title>
<style>
  body { margin: 0; background: #0b1220; color: #e6edf7;
         font-family: ui-sans-serif, system-ui, sans-serif; }
  .wrap { max-width: 860px; margin: 0 auto; padding: 24px 16px; }
  .frame-box { width: 100%; overflow-x: auto; background: #05070d; border-radius: 8px; }
  img#frame { display: block; width: 100%; height: auto; image-rendering: pixelated; border-radius: 8px; }
  .controls { display: flex; align-items: center; gap: 12px; margin-top: 12px; flex-wrap: wrap; }
  button { background: #35e0d0; color: #04121a; border: 0; border-radius: 8px;
           font: 650 0.9rem inherit; padding: 8px 16px; cursor: pointer; min-width: 84px; }
  button:hover { filter: brightness(1.08); }
  .step { color: #8ea0c0; font-size: 0.85rem; min-width: 96px; }
  .step b { color: #e6edf7; }
  input[type=range] { flex: 1; min-width: 160px; accent-color: #35e0d0; }
  select { background: #0c1526; color: #e6edf7; border: 1px solid #22304d;
           border-radius: 6px; padding: 3px 6px; font: inherit; }
  label.speed { color: #8ea0c0; font-size: 0.8rem; display: flex; align-items: center; gap: 6px; }
</style>
</head>
<body>
<div class="wrap">
  <div class="frame-box"><img id="frame" alt="environment frame"></div>
  <div class="controls">
    <button id="play">Pause</button>
    <span class="step">step <b id="stepnum">0</b> / __LAST__</span>
    <input type="range" id="scrub" min="0" max="__LAST__" value="0">
    <label class="speed">speed
      <select id="speed">
        <option value="2">0.5x</option>
        <option value="1" selected>1x</option>
        <option value="0.5">2x</option>
        <option value="0.25">4x</option>
      </select>
    </label>
  </div>
</div>
<script>
  const FRAMES = __FRAMES__;
  const BASE_INTERVAL = __INTERVAL__;
  const img = document.getElementById('frame');
  const scrub = document.getElementById('scrub');
  const stepnum = document.getElementById('stepnum');
  const playBtn = document.getElementById('play');
  const speedSel = document.getElementById('speed');
  let i = 0, timer = null;
  function show(idx) { i = idx; img.src = FRAMES[i]; scrub.value = i; stepnum.textContent = i; }
  function tick() { show((i + 1) % FRAMES.length); }
  function start() { stop(); timer = setInterval(tick, BASE_INTERVAL * parseFloat(speedSel.value)); playBtn.textContent = 'Pause'; }
  function stop() { if (timer) clearInterval(timer); timer = null; playBtn.textContent = 'Play'; }
  playBtn.onclick = () => timer ? stop() : start();
  scrub.oninput = () => { stop(); show(parseInt(scrub.value, 10)); };
  speedSel.onchange = () => { if (timer) start(); };
  if (FRAMES.length) { show(0); start(); }
</script>
</body>
</html>
"""


class ColabVideo():
    def __init__(self, path):
        # Source: https://stackoverflow.com/a/57378660/3890306
        self.video_src = 'data:video/mp4;base64,' + b64encode(open(path, 'rb').read()).decode()

    def _repr_html_(self):
        return """
        <video width=400 controls>
              <source src="{}" type="video/mp4">
        </video>
        """.format(self.video_src)


class ColabHTML():
    """Display a render_html() export inline in a notebook via a sandboxed iframe."""

    def __init__(self, path, width=440, height=380):
        with open(path) as f:
            self.doc = f.read()
        self.width, self.height = width, height

    def _repr_html_(self):
        from html import escape
        return '<iframe srcdoc="{}" width="{}" height="{}" style="border:0"></iframe>'.format(
            escape(self.doc, quote=True), self.width, self.height)
