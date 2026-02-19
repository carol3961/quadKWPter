import os
import re
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from PyFlyt.gym_envs import FlattenWaypointEnv
from quadx_forest_env import QuadXForestEnv

# =========================
# CONFIG
# =========================
EXP_NAME = "forest_obstacle_test"
RUN_ID = "run_1"  # <-- change to run_2, run_3, etc.

NUM_TREES = 0          # match what you want to visualize
NUM_TARGETS = 1
NUM_SENSORS = 8
SENSOR_RANGE = 5.0
CONTEXT_LENGTH = 1
N_STACK = 4            # MUST match training

# =========================
# Helpers
# =========================
def resolve_model_path(log_dir: str, run_id: str) -> str:
    """
    SB3 save() usually produces a .zip. We'll accept either.
    """
    base = os.path.join(log_dir, run_id, f"final_model_{run_id}")
    if os.path.exists(base):
        return base
    if os.path.exists(base + ".zip"):
        return base + ".zip"
    raise FileNotFoundError(f"Could not find model at {base}(.zip)")

def make_env():
    env = QuadXForestEnv(
        render_mode="human",
        num_trees=NUM_TREES,
        num_targets=NUM_TARGETS,
        num_sensors=NUM_SENSORS,
        sensor_range=SENSOR_RANGE,
    )
    env = FlattenWaypointEnv(env, context_length=CONTEXT_LENGTH)
    return env

# =========================
# Main
# =========================
if __name__ == "__main__":
    log_dir = os.path.join(".", "logs", EXP_NAME)
    run_dir = os.path.join(log_dir, RUN_ID)

    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # Build env with SAME wrappers as training
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=N_STACK)

    # Load VecNormalize stats (recommended)
    vecnormalize_path = os.path.join(run_dir, "vecnormalize.pkl")

    if os.path.exists(vecnormalize_path):
        print(f"Loading VecNormalize stats from: {vecnormalize_path}")
        env = VecNormalize.load(vecnormalize_path, env)
        env.training = False
        env.norm_reward = False  # show true rewards during viz
    else:
        print("Warning: VecNormalize stats not found. "
              "Continuing without saved stats (results may look worse).")
        env = VecNormalize(env, norm_obs=True, norm_reward=False, training=False)

    # Load model
    model_path = resolve_model_path(log_dir, RUN_ID)
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path, env=env)

    obs = env.reset()

    print("Running visualization... (CTRL+C to quit)")
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
    except KeyboardInterrupt:
        print("\nExiting visualization.")
    finally:
        env.close()


# # import gymnasium as gym
# # from PyFlyt.gym_envs import FlattenWaypointEnv
# # from stable_baselines3 import PPO
# # from quadx_forest_env import QuadXForestEnv

# # env = QuadXForestEnv(
# #     render_mode="human",
# #     num_trees=5,
# #     num_targets=1,
# #     num_sensors=8,
# #     sensor_range=5.0
# # )
# # env = FlattenWaypointEnv(env, context_length=1)

# # model = PPO.load("final_model_2026-02-17_18-02-22", env=env)

# # obs, _ = env.reset()

# # episodes = 0
# # successes = 0
# # crashes = 0

# # while True:
# #     action, _ = model.predict(obs, deterministic=True)
# #     obs, reward, terminated, truncated, info = env.step(action)
    
# #     if terminated or truncated:
# #         obs, _ = env.reset()
# import gymnasium as gym
# from PyFlyt.gym_envs import FlattenWaypointEnv
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
# from quadx_forest_env import QuadXForestEnv
# import os

# EXP_NAME = "forest_obstacle_avoidance_v3"
# RUN_ID = "2026-02-18_16-52-42"

# def make_env():
#     env = QuadXForestEnv(
#         render_mode="human",
#         num_trees=0,
#         num_targets=1,
#         num_sensors=8,
#         sensor_range=5.0
#     )
#     env = FlattenWaypointEnv(env, context_length=1)
#     return env

# # Create env with same wrappers as training
# env = DummyVecEnv([make_env])
# env = VecFrameStack(env, n_stack=4)  # <--- MISSING: Model was trained with frame stacking

# # Try to load VecNormalize stats if they exist
# vecnormalize_path = f"./logs/{EXP_NAME}/{RUN_ID}/final_model_{RUN_ID}.pkl"  
# if os.path.exists(vecnormalize_path):
#     print(f"Loading VecNormalize stats from {vecnormalize_path}")
#     env = VecNormalize.load(vecnormalize_path, env)
#     env.training = False  # Don't update stats during testing
#     env.norm_reward = False  # Don't normalize rewards during testing
# else:
#     print("Warning: VecNormalize stats not found, wrapping anyway")
#     env = VecNormalize(env, norm_obs=True, norm_reward=False, training=False)

# # Load model
# model = PPO.load(f"./logs/{EXP_NAME}/{RUN_ID}/final_model_{RUN_ID}", env=env)

# obs = env.reset()

# episodes = 0
# successes = 0
# crashes = 0

# print("Running evaluation...")

# while True:
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
    
#     if done:  
#         obs = env.reset()