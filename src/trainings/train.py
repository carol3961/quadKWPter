import os
import re
import shutil
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback
from PyFlyt.gym_envs import FlattenWaypointEnv
from quadx_forest_env import QuadXForestEnv
import imageio_ffmpeg
from tensorboard_video_recorder import TensorboardVideoRecorder

os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()

# =========================
# CONFIG
# =========================

# --------------Hyperparameters for Training---------------
NUM_ENVS = 8
NUM_SENSORS = 8
NUM_TREES = 5
TOTAL_TIMESTEPS = 100_000
N_STEPS = 2048
BATCH_SIZE = 1024
LEARNING_RATE = 1e-4
N_EPOCHS = 10
GAMMA = 0.99
ENT_COEF = 0.1

# --------------Run / checkpoint management settings---------------
# If True: resumes the most recent run_N (keeps training inside that same run folder).
# If False: starts a brand new run_(N+1) folder (unless forking, see below).
RESUME_LATEST_RUN = False
# EXP_NAME = "forest_obstacle_avoidance_v5"
EXP_NAME = "test"
CHECKPOINT_SAVE_FREQ = 50_000
N_STACK = 2

# If START_FROM_RUN is not None, we will create a NEW run_N directory and initialize it from
# a checkpoint in START_FROM_RUN (optionally at START_FROM_STEPS).
START_FROM_RUN = None          # e.g. "run_1" or None
START_FROM_STEPS =  None       # e.g. 100000 (int) or None -> uses latest checkpoint in START_FROM_RUN
COPY_VECNORM_ON_FORK = True

# ----------- Video recording (TensorBoard) -----------------------
VID_LEN = 2000
FPS = 30
VIDEO_EVERY_STEPS = 2000   # record when step % VIDEO_EVERY_STEPS == 0
RECORD_ENV_IDX = 0         # which env in the VecEnv to record
RECORD_VIDEO = False   # <--- toggle this

# =========================
# HELPERS
# =========================
def next_run_id(base_log_dir: str) -> str:
    if not os.path.isdir(base_log_dir):
        return "run_1"
    run_nums = []
    for name in os.listdir(base_log_dir):
        m = re.match(r"run_(\d+)$", name)
        if m and os.path.isdir(os.path.join(base_log_dir, name)):
            run_nums.append(int(m.group(1)))
    return f"run_{(max(run_nums) + 1) if run_nums else 1}"

def latest_run_dir(base_log_dir: str):
    if not os.path.isdir(base_log_dir):
        return None
    best_n, best_path = -1, None
    for name in os.listdir(base_log_dir):
        m = re.match(r"run_(\d+)$", name)
        if m:
            n = int(m.group(1))
            p = os.path.join(base_log_dir, name)
            if os.path.isdir(p) and n > best_n:
                best_n, best_path = n, p
    return best_path

def latest_checkpoint(ckpt_dir: str, prefix: str = "checkpoint"):
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

def checkpoint_path_for_steps(run_dir: str, steps: int, prefix: str = "checkpoint"):
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    path = os.path.join(ckpt_dir, f"{prefix}_{steps}_steps.zip")
    return path if os.path.exists(path) else None

def make_env():
    env = QuadXForestEnv(
        render_mode="rgb_array" if RECORD_VIDEO else None,
        num_trees=NUM_TREES,
        num_targets=1,
        num_sensors=NUM_SENSORS,
        sensor_range=5.0,
        max_duration_seconds=30.0,
        flight_dome_size=12.0
    )
    return FlattenWaypointEnv(env, context_length=1)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    base_log_dir = os.path.join(".", "logs", EXP_NAME)
    os.makedirs(base_log_dir, exist_ok=True)

    # ----- Decide target run folder -----
    # Forking always creates a NEW run folder, even if RESUME_LATEST_RUN=True.
    if START_FROM_RUN is not None:
        run_id = next_run_id(base_log_dir)
        log_dir = os.path.join(base_log_dir, run_id)
        print(f"Starting NEW forked run: {run_id} (from {START_FROM_RUN})")
    else:
        if RESUME_LATEST_RUN:
            prev = latest_run_dir(base_log_dir)
            if prev:
                run_id = os.path.basename(prev)
                log_dir = prev
                print(f"✓ Resuming latest run: {run_id}")
            else:
                run_id = next_run_id(base_log_dir)
                log_dir = os.path.join(base_log_dir, run_id)
                print(f"✗ No prior runs. Starting fresh: {run_id}")
        else:
            run_id = next_run_id(base_log_dir)
            log_dir = os.path.join(base_log_dir, run_id)
            print(f"Starting fresh: {run_id}")

    os.makedirs(log_dir, exist_ok=True)

    # Per-run checkpoint folder
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Per-run VecNormalize stats file
    vecnorm_path = os.path.join(log_dir, "vecnormalize.pkl")

    # ----- Build base env + wrappers that must always match -----
    env = make_vec_env(make_env, n_envs=NUM_ENVS, vec_env_cls=SubprocVecEnv)
    env = VecFrameStack(env, n_stack=N_STACK)

    if RECORD_VIDEO:
        video_trigger = lambda step: step % VIDEO_EVERY_STEPS == 0
        env = TensorboardVideoRecorder(
            env=env,
            video_trigger=video_trigger,
            video_length=VID_LEN,
            fps=FPS,
            record_video_env_idx=RECORD_ENV_IDX,
            tb_log_dir=log_dir,  # keep everything inside the run folder
        )

    # ----- Model / VecNormalize initialization -----
    model = None
    reset_timesteps = True

    if START_FROM_RUN is not None:
        # ===== Fork from a specific run (and optionally specific checkpoint steps) =====
        source_run_dir = os.path.join(base_log_dir, START_FROM_RUN)
        if not os.path.isdir(source_run_dir):
            raise FileNotFoundError(f"START_FROM_RUN not found: {source_run_dir}")

        source_vecnorm = os.path.join(source_run_dir, "vecnormalize.pkl")
        if not os.path.exists(source_vecnorm):
            raise FileNotFoundError(f"VecNormalize stats not found in source run: {source_vecnorm}")

        # Choose checkpoint
        if START_FROM_STEPS is not None:
            source_ckpt = checkpoint_path_for_steps(source_run_dir, START_FROM_STEPS, prefix="checkpoint")
            if source_ckpt is None:
                raise FileNotFoundError(
                    f"Checkpoint not found for steps={START_FROM_STEPS} in {os.path.join(source_run_dir, 'checkpoints')}"
                )
        else:
            source_ckpt = latest_checkpoint(os.path.join(source_run_dir, "checkpoints"), prefix="checkpoint")
            if source_ckpt is None:
                raise FileNotFoundError(f"No checkpoints found in source run: {os.path.join(source_run_dir, 'checkpoints')}")

        print(f"✓ Forking from {START_FROM_RUN} checkpoint: {source_ckpt}")
        print(f"✓ Loading VecNormalize stats from source: {source_vecnorm}")

        # Load VecNormalize stats from source and attach to current env
        env = VecNormalize.load(source_vecnorm, env)
        env.training = True
        env.norm_reward = True

        # Optionally copy vecnorm file into this new run folder (self-contained)
        if COPY_VECNORM_ON_FORK:
            shutil.copy2(source_vecnorm, vecnorm_path)
            print(f"✓ Copied VecNormalize stats into new run: {vecnorm_path}")

        # Load model from the chosen source checkpoint
        model = PPO.load(source_ckpt, env=env, tensorboard_log=log_dir)
        reset_timesteps = False

    else:
        # ===== Normal (non-fork) behavior: load vecnorm from this run if exists, else create new =====
        if os.path.exists(vecnorm_path):
            print(f"✓ Loading VecNormalize stats: {vecnorm_path}")
            env = VecNormalize.load(vecnorm_path, env)
            env.training = True
            env.norm_reward = True
        else:
            print("✗ No VecNormalize stats found. Creating new VecNormalize.")
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

        # Resume from latest checkpoint in this run (if any)
        load_path = latest_checkpoint(ckpt_dir, prefix="checkpoint")

        if load_path:
            print(f"✓ Resuming from checkpoint: {load_path}")
            model = PPO.load(load_path, env=env, tensorboard_log=log_dir)
            reset_timesteps = False
        else:
            print("✗ No checkpoint found in this run. Starting PPO fresh.")
            model = PPO(
                "MlpPolicy",
                env,
                verbose=0,
                tensorboard_log=log_dir,
                n_steps=N_STEPS,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                n_epochs=N_EPOCHS,
                gamma=GAMMA,
                ent_coef=ENT_COEF
            )
            reset_timesteps = True

    # ----- Checkpointing -----
    checkpoint_callback = CheckpointCallback(
        # save_freq is in "calls" (per env step), so divide by NUM_ENVS for approx env-steps frequency
        save_freq=max(1, CHECKPOINT_SAVE_FREQ // NUM_ENVS),
        save_path=ckpt_dir,
        name_prefix="checkpoint"
    )

    # ----- Train -----
    print(f"Training for {TOTAL_TIMESTEPS:,} timesteps...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        reset_num_timesteps=reset_timesteps,
        progress_bar=False
    )

    # ----- Save final artifacts (self-contained run folder) -----
    env.save(vecnorm_path)
    model.save(os.path.join(log_dir, f"final_model_{run_id}"))
    env.close()

    print("Training complete!")
