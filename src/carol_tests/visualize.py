import gymnasium as gym
from PyFlyt.gym_envs import FlattenWaypointEnv
from stable_baselines3 import PPO
from quadx_forest_env import QuadXForestEnv

env = QuadXForestEnv(
    render_mode="human",
    num_trees=5,
    num_targets=1,
    num_sensors=8,
    sensor_range=5.0
)
env = FlattenWaypointEnv(env, context_length=1)

model = PPO.load("quadx_forest_avoidance", env=env)

obs, _ = env.reset()

episodes = 0
successes = 0
crashes = 0

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, _ = env.reset()