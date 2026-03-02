"""
Cluster training script: same as your current script but with timestep-based
checkpoints so files are saved every CHECKPOINT_SAVE_FREQ steps (10k, 20k, ...).
Replace your checkpoint section with the one below, and add the callback class + import.
"""
import os
import re
import shutil
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
from PyFlyt.gym_envs import FlattenWaypointEnv
from quadx_forest_env import QuadXForestEnv


# =========================
# CONFIG
# =========================
NUM_ENVS = 8
BATCH_SIZE = 64
TOTAL_TIMESTEPS = 1_000_000
run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
EXP_NAME = f"forest_obstacle_test_{NUM_ENVS}E_{BATCH_SIZE}BS_{run_id}"

RESUME_LATEST_RUN = False
CHECKPOINT_SAVE_FREQ = 10_000
N_STACK = 4
START_FROM_RUN = None
START_FROM_STEPS = None
COPY_VECNORM_ON_FORK = True


# =========================
# CHECKPOINT CALLBACK (saves every N env steps, not every N callback calls)
# =========================
class CheckpointEveryNSteps(BaseCallback):
    """Saves a checkpoint every N environment steps (num_timesteps). Optionally saves VecNormalize too."""
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "checkpoint", vecnorm_path: str = None, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.vecnorm_path = vecnorm_path
        self._last_saved = 0

    def _init_callback(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_saved >= self.save_freq:
            self._last_saved = (self.num_timesteps // self.save_freq) * self.save_freq
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self._last_saved}_steps.zip")
            self.model.save(path)
            vec_env = self.model.get_vec_normalize_env()
            if vec_env is not None and self.vecnorm_path:
                vec_env.save(self.vecnorm_path)
            if self.verbose:
                print(f"Checkpoint saved: {path}")
        return True


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
        render_mode=None,
        num_trees=0,
        num_targets=1,
        num_sensors=8,
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

    if START_FROM_RUN is not None:
        run_id_inner = next_run_id(base_log_dir)
        log_dir = os.path.join(base_log_dir, run_id_inner)
        print(f"Starting NEW forked run: {run_id_inner} (from {START_FROM_RUN})")
    else:
        if RESUME_LATEST_RUN:
            prev = latest_run_dir(base_log_dir)
            if prev:
                run_id_inner = os.path.basename(prev)
                log_dir = prev
                print(f"✓ Resuming latest run: {run_id_inner}")
            else:
                run_id_inner = next_run_id(base_log_dir)
                log_dir = os.path.join(base_log_dir, run_id_inner)
                print(f"✗ No prior runs. Starting fresh: {run_id_inner}")
        else:
            run_id_inner = next_run_id(base_log_dir)
            log_dir = os.path.join(base_log_dir, run_id_inner)
            print(f"Starting fresh: {run_id_inner}")

    os.makedirs(log_dir, exist_ok=True)
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    vecnorm_path = os.path.join(log_dir, "vecnormalize.pkl")

    env = make_vec_env(make_env, n_envs=NUM_ENVS, vec_env_cls=SubprocVecEnv)
    env = VecFrameStack(env, n_stack=N_STACK)

    model = None
    reset_timesteps = True

    if START_FROM_RUN is not None:
        source_run_dir = os.path.join(base_log_dir, START_FROM_RUN)
        if not os.path.isdir(source_run_dir):
            raise FileNotFoundError(f"START_FROM_RUN not found: {source_run_dir}")
        source_vecnorm = os.path.join(source_run_dir, "vecnormalize.pkl")
        if not os.path.exists(source_vecnorm):
            raise FileNotFoundError(f"VecNormalize stats not found in source run: {source_vecnorm}")
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
        env = VecNormalize.load(source_vecnorm, env)
        env.training = True
        env.norm_reward = True
        if COPY_VECNORM_ON_FORK:
            shutil.copy2(source_vecnorm, vecnorm_path)
            print(f"✓ Copied VecNormalize stats into new run: {vecnorm_path}")
        model = PPO.load(source_ckpt, env=env, tensorboard_log=log_dir)
        reset_timesteps = False
    else:
        if os.path.exists(vecnorm_path):
            print(f"✓ Loading VecNormalize stats: {vecnorm_path}")
            env = VecNormalize.load(vecnorm_path, env)
            env.training = True
            env.norm_reward = True
        else:
            print("✗ No VecNormalize stats found. Creating new VecNormalize.")
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
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
                n_steps=2048 // NUM_ENVS,
                batch_size=BATCH_SIZE,
                learning_rate=3e-4,
                n_epochs=10,
                gamma=0.99
            )
            reset_timesteps = True

    # ----- Checkpointing: every CHECKPOINT_SAVE_FREQ env steps (10k, 20k, ...) -----
    checkpoint_callback = CheckpointEveryNSteps(
        save_freq=CHECKPOINT_SAVE_FREQ,
        save_path=ckpt_dir,
        name_prefix="checkpoint",
        vecnorm_path=vecnorm_path,
    )

    print(f"Training for {TOTAL_TIMESTEPS:,} timesteps...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        reset_num_timesteps=reset_timesteps,
        progress_bar=False,
        tb_log_name=run_id_inner,
    )

    env.save(vecnorm_path)
    model.save(os.path.join(log_dir, f"final_model_{run_id_inner}"))
    env.close()
    print("Training complete!")
