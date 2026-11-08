import os
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

ENV_ID      = "Humanoid-v5"
N_ENVS      = 8                
TOTAL_STEPS = 15_000_000       
XML = os.path.abspath("humanoid2.xml")   

## penalty to fix crouching problem
class UprightTermination(gym.Wrapper):
    """End the episode if the pelvis tilts more than ~30 degrees from vertical."""
    def __init__(self, env, min_up=0.85):
        super().__init__(env)
        self.min_up = min_up
    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        if self.env.unwrapped.data.xmat[1][8] < self.min_up:   # 1.0 = upright
            terminated = True
        return obs, r, terminated, truncated, info

# penalty for turning away from x direction
class HeadingPenalty(gym.Wrapper):
    """Penalize facing away from +x (sin of the yaw angle, squared)."""
    def __init__(self, env, weight=2.0):
        super().__init__(env)
        self.weight = weight
    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        fwd = -self.env.unwrapped.data.xmat[1].reshape(3, 3)[:, 1]   # robot's forward axis in world
        r -= self.weight * fwd[1] ** 2
        return obs, r, terminated, truncated, info

def wrap(env):
    return HeadingPenalty(UprightTermination(env))
        
def main():
    os.makedirs("models", exist_ok=True)
    env = make_vec_env(
    ENV_ID, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv,
    env_kwargs=dict(xml_file=XML, healthy_z_range=(0.33, 0.70)),
    wrapper_class=wrap,
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy", env, verbose=1, tensorboard_log="./tb/",
        n_steps=512, batch_size=256, n_epochs=5,
        gamma=0.95, gae_lambda=0.9, clip_range=0.3,
        ent_coef=0.00238306, vf_coef=0.431892, max_grad_norm=2.0,
        learning_rate=3.56987e-5,
        policy_kwargs=dict(
            log_std_init=-2, ortho_init=False, activation_fn=nn.ReLU,
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        ),
    )
    ckpt = CheckpointCallback(save_freq=max(200_000 // N_ENVS, 1),
                              save_path="./models/", name_prefix=ENV_ID,
                              save_vecnormalize=True)
    model.learn(total_timesteps=TOTAL_STEPS, callback=ckpt, progress_bar=True)
    model.save(f"models/{ENV_ID}_final")
    env.save(f"models/{ENV_ID}_vecnormalize.pkl")
    print("Done. Watch with:  python watch_ppo.py")


if __name__ == "__main__":
    main()
