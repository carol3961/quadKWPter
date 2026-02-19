import os
import re
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback
from PyFlyt.gym_envs import FlattenWaypointEnv
from quadx_forest_env import QuadXForestEnv

NUM_ENVS = 8
TOTAL_TIMESTEPS = 50_000
# EXP_NAME = "forest_obstacle_avoidance_v1"
EXP_NAME = "forest_obstacle_test"
CHKPT_DIR = f"./checkpoints/{EXP_NAME}/"
# tensor command: tensorboard --logdir=logs/{exp_name}/{run_id}/PPO_1

def latest_checkpoint(ckpt_dir, prefix="checkpoint"):
    """Find most recent checkpoint"""
    if not os.path.isdir(ckpt_dir):
        return None
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)_steps\.zip$")
    best_path, best_steps = None, -1
    for f in os.listdir(ckpt_dir):
        m = pattern.match(f)
        if m:
            steps = int(m.group(1))
            if steps > best_steps:
                best_steps, best_path = steps, os.path.join(ckpt_dir, f)
    return best_path

def make_env():
    env = QuadXForestEnv(
        render_mode=None,  # No rendering on HPC
        num_trees=5,
        num_targets=1,
        num_sensors=8,
        sensor_range=5.0,
        max_duration_seconds=30.0,
        flight_dome_size=12.0
    )
    return FlattenWaypointEnv(env, context_length=1)

if __name__ == "__main__":
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"./logs/{EXP_NAME}/{run_id}"
    
    os.makedirs(CHKPT_DIR, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create vectorized environment with true parallelization
    env = make_vec_env(
        make_env, 
        n_envs=NUM_ENVS,
        vec_env_cls=SubprocVecEnv  # KEY: Use parallel processes
    )
    
    # Add frame stacking for temporal information
    env = VecFrameStack(env, n_stack=4)
    
    # Normalize observations and rewards
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    env.save(f"./logs/{EXP_NAME}/{run_id}/final_model_{run_id}.pkl")
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000 // NUM_ENVS,
        save_path=CHKPT_DIR,
        name_prefix="checkpoint"
    )
    
    # Try to resume from checkpoint
    load_path = latest_checkpoint(CHKPT_DIR, "checkpoint")
    
    if load_path:
        print(f"✓ Resuming from {load_path}")
        model = PPO.load(load_path, env=env, tensorboard_log=log_dir)
        reset_timesteps = False
    else:
        print("✗ No checkpoint found. Starting fresh.")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            tensorboard_log=log_dir,
            n_steps=2048 // NUM_ENVS,  # Scale with num envs
            batch_size=256,
            learning_rate=3e-4,
            n_epochs=10,
            gamma=0.99
        )
        reset_timesteps = True
    
    # Train
    print(f"Training for {TOTAL_TIMESTEPS:,} timesteps...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        reset_num_timesteps=reset_timesteps,
        progress_bar=False  # Cleaner logs on HPC
    )
    
    # Save final model
    model.save(f"./logs/{EXP_NAME}/{run_id}/final_model_{run_id}")
    env.close()
    
    print("Training complete!")