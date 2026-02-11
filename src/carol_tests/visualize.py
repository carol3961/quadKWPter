import gymnasium as gym
from PyFlyt.gym_envs import FlattenWaypointEnv
from stable_baselines3 import PPO
from quadx_forest_env import QuadXForestEnv

#env = gym.make("PyFlyt/QuadX-Waypoints-v4", render_mode="human")
env = QuadXForestEnv(
    render_mode="human",
    num_trees=10,
    tree_radius_range=(0.1, 0.3),
    tree_height_range=(2.0, 4.0)
)
env = FlattenWaypointEnv(env, context_length=2)

model = PPO.load("quadx_waypoints.zip", env=env)

obs, _ = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, _ = env.reset()
