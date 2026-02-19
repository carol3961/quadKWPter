# import gymnasium as gym
# from PyFlyt.gym_envs import FlattenWaypointEnv
# from stable_baselines3 import PPO
# from quadx_forest_env import QuadXForestEnv

# env = QuadXForestEnv(
#     render_mode="human",
#     num_trees=5,
#     num_targets=1,
#     num_sensors=8,
#     sensor_range=5.0
# )
# env = FlattenWaypointEnv(env, context_length=1)

# model = PPO.load("final_model_2026-02-17_18-02-22", env=env)

# obs, _ = env.reset()

# episodes = 0
# successes = 0
# crashes = 0

# while True:
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)
    
#     if terminated or truncated:
#         obs, _ = env.reset()
import gymnasium as gym
from PyFlyt.gym_envs import FlattenWaypointEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from quadx_forest_env import QuadXForestEnv
import os

EXP_NAME = "forest_obstacle_test"
RUN_ID = "2026-02-18_16-21-12"

def make_env():
    env = QuadXForestEnv(
        render_mode="human",
        num_trees=5,
        num_targets=1,
        num_sensors=8,
        sensor_range=5.0
    )
    env = FlattenWaypointEnv(env, context_length=1)
    return env

# Create env with same wrappers as training
env = DummyVecEnv([make_env])
env = VecFrameStack(env, n_stack=4)  # <--- MISSING: Model was trained with frame stacking

# Try to load VecNormalize stats if they exist
vecnormalize_path = f"./logs/{EXP_NAME}/{RUN_ID}/final_model_{RUN_ID}.pkl"  
if os.path.exists(vecnormalize_path):
    print(f"Loading VecNormalize stats from {vecnormalize_path}")
    env = VecNormalize.load(vecnormalize_path, env)
    env.training = False  # Don't update stats during testing
    env.norm_reward = False  # Don't normalize rewards during testing
else:
    print("Warning: VecNormalize stats not found, wrapping anyway")
    env = VecNormalize(env, norm_obs=True, norm_reward=False, training=False)

# Load model
model = PPO.load(f"./logs/{EXP_NAME}/{RUN_ID}/final_model_{RUN_ID}", env=env)

obs = env.reset()

episodes = 0
successes = 0
crashes = 0

print("Running evaluation...")

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    
    if done:  
        obs = env.reset()