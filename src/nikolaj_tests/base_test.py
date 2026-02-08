import os
import re
import gymnasium as gym
import PyFlyt.gym_envs  # registers envs
from PyFlyt.gym_envs import FlattenWaypointEnv

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from tensorboard_video_recorder import TensorboardVideoRecorder
from datetime import datetime
experiment_name = "cluster_run"
run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
experiment_logdir = f"quadKWPter_logs/{experiment_name}/{run_id}"

checkpoint_dir = f"./checkpoints/{experiment_name}/"
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_prefix = "quadx_checkpoints"


env = gym.make("PyFlyt/QuadX-Waypoints-v4", render_mode="rgb_array")
env = FlattenWaypointEnv(env, context_length=2)
env = Monitor(env)

video_trigger = lambda step: step % 2000 == 0
env = TensorboardVideoRecorder(
    env=env,
    video_trigger=video_trigger,
    video_length=2000,
    fps=30,
    record_video_env_idx=0,
    tb_log_dir=experiment_logdir,
)

checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path=checkpoint_dir,
    name_prefix=checkpoint_prefix,
)

def latest_checkpoint_path(ckpt_dir: str, prefix: str) -> str | None:
    if not os.path.isdir(ckpt_dir):
        return None
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)_steps\.zip$")
    best = None
    best_steps = -1
    for f in os.listdir(ckpt_dir):
        m = pattern.match(f)
        if m:
            steps = int(m.group(1))
            if steps > best_steps:
                best_steps = steps
                best = os.path.join(ckpt_dir, f)
    return best

load_path = latest_checkpoint_path(checkpoint_dir, checkpoint_prefix)

if load_path:
    print(f"FOUND CHECKPOINT -- Resuming from {load_path}")
    model = PPO.load(load_path, env=env)
    reset_timesteps = False
else:
    print("No checkpoint found. Starting new training.")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=experiment_logdir,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
    )
    reset_timesteps = True

model.learn(
    total_timesteps=1_000_000,
    tb_log_name="quadx_waypoints",
    callback=checkpoint_callback,
    reset_num_timesteps=reset_timesteps,
)

model.save("quadx_waypoints_gpu")
env.close()
