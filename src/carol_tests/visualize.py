from quadx_forest_env import QuadXForestEnv
from PyFlyt.gym_envs import FlattenWaypointEnv
from stable_baselines3 import PPO

env = QuadXForestEnv(
    render_mode="human",
    num_trees=15,
    num_targets=1,
    num_sensors=16,
    sensor_range=5.0
)
env = FlattenWaypointEnv(env, context_length=1)

model = PPO.load("quadx_waypoints.zip", env=env)

obs, _ = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, _ = env.reset()
