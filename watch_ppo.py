#   python3 watch_ppo.py 600000          -> to view checkpoint at 600k steps (+ its matching stats file)
import os
import re
import sys
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ENV_ID = "Humanoid-v5"


class UprightTermination(gym.Wrapper):
    """Must match train_ppo.py: end the episode if the pelvis tilts more than ~30 degrees."""
    def __init__(self, env, min_up=0.85):
        super().__init__(env)
        self.min_up = min_up

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        if self.env.unwrapped.data.xmat[1][8] < self.min_up:   # 1.0 = upright
            terminated = True
        return obs, r, terminated, truncated, info


def resolve(arg):
    """Return (model_path, stats_path) for a step count, 'final', or an explicit path."""
    if arg is None or arg == "final":
        return f"models/{ENV_ID}_final.zip", f"models/{ENV_ID}_vecnormalize.pkl"
    if arg.isdigit():
        return (f"models/{ENV_ID}_{arg}_steps.zip",
                f"models/{ENV_ID}_vecnormalize_{arg}_steps.pkl")
    folder = os.path.dirname(arg) or "."
    m = re.search(r"_(\d+)_steps\.zip$", arg)
    stats = (f"{ENV_ID}_vecnormalize_{m.group(1)}_steps.pkl" if m
             else f"{ENV_ID}_vecnormalize.pkl")
    return arg, os.path.join(folder, stats)


model_path, stats_path = resolve(sys.argv[1] if len(sys.argv) > 1 else None)

# Use the exact same custom model file and parameters used in training
XML = os.path.abspath("humanoid2.xml")
env_kwargs = {
    "render_mode": "human",
    "xml_file": XML,
    "healthy_z_range": (0.33, 0.70),
}

env = DummyVecEnv([lambda: UprightTermination(gym.make(ENV_ID, **env_kwargs))])

try:
    env = VecNormalize.load(stats_path, env)
    env.training = False       # Stop updating normalization stats during playback
    env.norm_reward = False    # Do not normalize rewards during playback
    print(f"Loaded normalization stats from {stats_path}")
except FileNotFoundError:
    print(f"WARNING: {stats_path} not found. Running UNNORMALIZED - the robot will probably "
          f"act randomly because the policy was trained on normalized observations.")

model = PPO.load(model_path, env=env)
print(f"Loaded {model_path}  -  ESC or Ctrl-C to quit")

obs = env.reset()
try:
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = env.step(action)   # DummyVecEnv resets finished episodes by itself
except KeyboardInterrupt:
    print("\nClosed.")
finally:
    env.close()
