import gymnasium as gym
import PyFlyt.gym_envs
from PyFlyt.gym_envs import FlattenWaypointEnv
from stable_baselines3 import PPO

env = gym.make("PyFlyt/QuadX-Waypoints-v4", render_mode="human")
env = FlattenWaypointEnv(env, context_length=2)

model = PPO.load("quadx_waypoints.zip", env=env)

obs, _ = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, _ = env.reset()
