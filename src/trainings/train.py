import os
import re
import shutil
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
from PyFlyt.gym_envs import FlattenWaypointEnv
from quadx_forest_env import QuadXForestEnv
import imageio_ffmpeg
from tensorboard_video_recorder import TensorboardVideoRecorder
from wandb.integration.sb3 import WandbCallback
import wandb
import numpy as np

os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()

# =========================
# CONFIG
# =========================

# --------------Hyperparameters for Training---------------
NUM_ENVS = 8
NUM_SENSORS = 8
NUM_TREES = 15
TOTAL_TIMESTEPS = 1_000_000
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
EXP_NAME = "forest_obstacle_avoidance_v7"
# EXP_NAME = "test"
CHECKPOINT_SAVE_FREQ = 50_000
N_STACK = 2

# If START_FROM_RUN is not None, we will create a NEW run_N directory and initialize it from
# a checkpoint in START_FROM_RUN (optionally at START_FROM_STEPS).
START_FROM_RUN = None       # e.g. "run_1" or None
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
        flight_dome_size=12.0,
        goal_reach_distance=0.5 # CHANGED
    )
    return FlattenWaypointEnv(env, context_length=1)


class ForestEnvWandbCallback(BaseCallback):
    """Logs env-specific metrics to wandb: targets reached, tree/ground collisions, timeouts, failures, time to target."""

    def __init__(self, log_freq: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = max(1, log_freq)
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.num_targets_reached: list[int] = []
        self.env_complete: list[int] = []
        self.episodes_ended: int = 0
        # Failure / outcome counts per ended episode
        self.tree_collisions: list[int] = []
        self.ground_collisions: list[int] = []
        self.timeouts: list[int] = []
        self.out_of_bounds: list[int] = []
        # Time (steps) to reach target when episode ended with goal
        self.time_to_reach_target_steps: list[int] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", np.array([]))
        if not infos:
            return True
        for i, info in enumerate(infos):
            if not (i < len(dones) and dones[i]) and "episode" not in info:
                continue
            self.episodes_ended += 1
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"].get("l", 0))
            if "num_targets_reached" in info:
                self.num_targets_reached.append(info["num_targets_reached"])
            if info.get("env_complete"):
                self.env_complete.append(1)
                if "episode" in info:
                    self.time_to_reach_target_steps.append(info["episode"].get("l", 0))
            if info.get("tree_collision"):
                self.tree_collisions.append(1)
            if info.get("collision") and not info.get("tree_collision"):
                self.ground_collisions.append(1)
            if info.get("episode_timeout"):
                self.timeouts.append(1)
            if info.get("out_of_bounds"):
                self.out_of_bounds.append(1)
        return True

    def _on_rollout_end(self) -> None:
        if self.n_calls % self.log_freq != 0:
            return
        goal_count = sum(self.env_complete)

        log_dict = {
            "env/episode_reward_mean": float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0,
            "env/episode_reward_std": float(np.std(self.episode_rewards)) if self.episode_rewards else 0.0,
            "env/episode_length_mean": float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0,
            "env/num_targets_reached_mean": float(np.mean(self.num_targets_reached)) if self.num_targets_reached else 0.0,
            "env/goal_reached_per_rollout": goal_count,
            "env/goal_reached_rate": (goal_count / self.episodes_ended) if self.episodes_ended > 0 else 0.0,
            "env/time_to_reach_target_steps_mean": float(np.mean(self.time_to_reach_target_steps)) if self.time_to_reach_target_steps else 0.0,
            "env/tree_collision_count": sum(self.tree_collisions),
            "env/ground_collision_count": sum(self.ground_collisions),
            "env/timeout_count": sum(self.timeouts),
            "env/out_of_bounds_count": sum(self.out_of_bounds),
            "env/episodes_ended_this_rollout": self.episodes_ended,
        }

        wandb.log(log_dict)

        self.episode_rewards.clear()
        self.episode_lengths.clear()
        self.num_targets_reached.clear()
        self.env_complete.clear()
        self.time_to_reach_target_steps.clear()
        self.tree_collisions.clear()
        self.ground_collisions.clear()
        self.timeouts.clear()
        self.out_of_bounds.clear()
        self.episodes_ended = 0


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

    # ----- Wandb: config + TensorBoard sync for clear graphs -----
    config = {
        "total_timesteps": TOTAL_TIMESTEPS,
        "n_steps": N_STEPS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "n_epochs": N_EPOCHS,
        "gamma": GAMMA,
        "ent_coef": ENT_COEF,
        "num_envs": NUM_ENVS,
        "num_trees": NUM_TREES,
        "num_sensors": NUM_SENSORS,
        "n_stack": N_STACK,
        "exp_name": EXP_NAME,
        "run_id": run_id,
    }
    wandb.init(
        project="quadx-forest-obstacle-avoidance",
        entity="nperroch-uci",
        name=f"{EXP_NAME}_{run_id}",
        tags=[f"{EXP_NAME}", run_id],
        config=config,
        sync_tensorboard=True,
    )
    # TensorBoard path for wandb: SB3 writes events here; wandb syncs this dir when sync_tensorboard=True.
    # Use absolute path so it's correct regardless of CWD (e.g. on cluster).
    tb_log_dir = os.path.abspath(wandb.run.dir)
    print(f"TensorBoard log dir (for wandb): {tb_log_dir}")

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
        model = PPO.load(source_ckpt, env=env, tensorboard_log=tb_log_dir)
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
            model = PPO.load(load_path, env=env, tensorboard_log=tb_log_dir)
            reset_timesteps = False
        else:
            print("✗ No checkpoint found in this run. Starting PPO fresh.")
            model = PPO(
                "MlpPolicy",
                env,
                verbose=0,
                tensorboard_log=tb_log_dir,
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
        callback=[
            checkpoint_callback,
            ForestEnvWandbCallback(log_freq=1),
            WandbCallback(
                gradient_save_freq=0,
                model_save_freq=0,
                model_save_path=log_dir,
            ),
        ],
        reset_num_timesteps=reset_timesteps,
        progress_bar=False
    )

    # ----- Save final artifacts (self-contained run folder) -----
    env.save(vecnorm_path)
    model.save(os.path.join(log_dir, f"final_model_{run_id}"))
    env.close()
