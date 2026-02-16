import os
import re
import gymnasium as gym
import PyFlyt.gym_envs  # registers envs
from PyFlyt.gym_envs import FlattenWaypointEnv
from quadx_forest_env import QuadXForestEnv
import imageio_ffmpeg
os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from tensorboard_video_recorder import TensorboardVideoRecorder
from datetime import datetime

NUM_ENVS = 8
VID_LEN = 2000
FPS = 30
TOTAL_TIMESTEPS = 1_000_000
SEED = 0

EXP_NAME = "cluster_run_parallelized_v2_fewer_steps_before_update_bigger_batch_size"
CHKPT_DIR = f"./checkpoints/{EXP_NAME}/"
VECENV_MON_DIR = f"./vecenv_monitor_dir/{EXP_NAME}/"

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

def make_env():
    #env = gym.make("PyFlyt/QuadX-Waypoints-v4", render_mode="rgb_array")
    # env = gym.make("PyFlyt/QuadX-Waypoints-v4", render_mode=None)
    env = QuadXForestEnv(
        num_trees=5,
        num_targets=1,
        num_sensors=8,
        sensor_range=5.0,
        max_duration_seconds=30.0,
        flight_dome_size=12.0,
        render_mode="rgb_array"
    )
    env = FlattenWaypointEnv(env, context_length=2)
    return env

if __name__ == "__main__":

    # config
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_logdir = f"quadKWPter_logs/{EXP_NAME}/{run_id}"

    os.makedirs(CHKPT_DIR, exist_ok=True)
    os.makedirs(VECENV_MON_DIR, exist_ok=True)

    checkpoint_prefix = "quadx_checkpoints"

    env = make_vec_env(make_env, n_envs=NUM_ENVS,
                       seed=SEED,
                       monitor_dir=VECENV_MON_DIR,
                       vec_env_cls=SubprocVecEnv
                       )

    env = VecFrameStack(env, n_stack=4)

    video_trigger = lambda step: step % 2000 == 0
    env = TensorboardVideoRecorder(
        env=env,
        video_trigger=video_trigger,
        video_length=VID_LEN,
        fps=FPS,
        record_video_env_idx=0,
        tb_log_dir=experiment_logdir,
    )

    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000 // NUM_ENVS,
        save_path=CHKPT_DIR,
        name_prefix=checkpoint_prefix,
    )

    load_path = latest_checkpoint_path(CHKPT_DIR, checkpoint_prefix)

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
            n_steps=2048 // NUM_ENVS,
            batch_size=256,
            learning_rate=3e-4,
        )
        reset_timesteps = True

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        tb_log_name="quadx_waypoints",
        callback=checkpoint_callback,
        reset_num_timesteps=reset_timesteps,
    )

    model.save("quadx_waypoints_gpu")
    env.close()
