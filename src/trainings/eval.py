"""
evaluate.py  —  Fixed test-set evaluator for QuadX forest obstacle avoidance.

Creates 100 deterministic worlds (seeds 0-99) and measures:
  - Success rate
  - Failure breakdown (tree collision / floor crash / out-of-bounds / timeout)
  - Steps to completion (successes only)
  - Path efficiency  =  actual path length / straight-line distance  (1.0 = perfect)
  - Closest obstacle approach across the episode
  - Total (true) reward

Results are written to TensorBoard, x-axis = model.num_timesteps so that
running the evaluator after each training run lets you plot improvement over time.

Usage:
    python eval.py --exp forest_obstacle_avoidance_v6 --run run_2
"""

import os
import sys
import argparse
import pickle
import numpy as np
from collections import deque
from datetime import datetime

from stable_baselines3 import PPO
from PyFlyt.gym_envs import FlattenWaypointEnv
from torch.utils.tensorboard import SummaryWriter

from quadx_forest_env import QuadXForestEnv
from train import NUM_TREES, N_STACK, EXP_NAME, NUM_SENSORS, FLIGHT_DOME_SIZE

# =========================
# CONFIG  (edit or override via CLI)
# =========================
RUN_ID          = "run_1"        # which run folder to evaluate
NUM_TEST_WORLDS = 200
TEST_SEEDS      = list(range(NUM_TEST_WORLDS))   # seeds 0-99, fixed forever

NUM_TARGETS     = 1
SENSOR_RANGE    = 5.0
CONTEXT_LENGTH  = 1

# =========================
# HELPERS
# =========================

def load_vecnormalize_stats(vecnorm_path: str):
    """
    Load VecNormalize directly via pickle to extract obs_rms stats
    without spinning up a dummy physics environment.
    Returns the VecNormalize object (used only for its .obs_rms attribute).
    """
    with open(vecnorm_path, "rb") as f:
        vn = pickle.load(f)
    return vn


def normalize_obs(stacked_obs: np.ndarray, vn) -> np.ndarray:
    """
    Replicate what VecNormalize does during training:
        normalised = clip((x - mean) / sqrt(var + eps), -clip_obs, clip_obs)

    vn  : a VecNormalize object loaded via pickle (only .obs_rms etc. are used)
    """
    if not getattr(vn, "norm_obs", True):
        return stacked_obs
    obs = (stacked_obs - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + vn.epsilon)
    return np.clip(obs, -vn.clip_obs, vn.clip_obs).astype(np.float32)


def resolve_model_path(run_dir: str, run_id: str) -> str:
    """Model is saved inside the run folder: logs/exp/run_N/final_model_run_N(.zip)"""
    base = os.path.join(run_dir, f"final_model_{run_id}")
    for path in (base, base + ".zip"):
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No model found at {base}(.zip)")


def make_eval_env():
    env = QuadXForestEnv(
        render_mode=None,           # headless — faster
        num_trees=NUM_TREES,
        num_targets=NUM_TARGETS,
        num_sensors=NUM_SENSORS,
        sensor_range=SENSOR_RANGE,
        max_duration_seconds=30.0,
        flight_dome_size=FLIGHT_DOME_SIZE,
        goal_reach_distance=0.5,
    )
    return FlattenWaypointEnv(env, context_length=CONTEXT_LENGTH)


# =========================
# EPISODE RUNNER
# =========================

def run_episode(env, model, vn, n_stack: int, seed: int) -> dict:
    """
    Run one episode with the given seed and return a metrics dict.

    Frame-stacking and obs normalisation are handled manually here so that
    we can use a plain (non-VecEnv) environment and pass seeds directly.
    """
    obs_flat, _ = env.reset(seed=seed)

    # Initialise frame buffer: n_stack copies of the first obs
    frame_buffer = deque([obs_flat.copy() for _ in range(n_stack)], maxlen=n_stack)

    # Grab start / goal positions for path-efficiency metric
    inner: QuadXForestEnv = env.env
    _, _, _, lin_pos, _ = inner.compute_attitude()
    start_pos  = np.array(lin_pos).flatten().copy()
    goal_pos   = np.array(inner.waypoints.targets[0]).flatten().copy()
    straight_d = np.linalg.norm(start_pos - goal_pos)

    prev_pos          = start_pos.copy()
    path_length       = 0.0
    min_obstacle_dist = float("inf")
    total_reward      = 0.0
    steps             = 0
    success           = False
    failure_mode      = None

    while True:
        # Build normalised, stacked observation
        stacked = np.concatenate(list(frame_buffer), axis=0)
        normed  = normalize_obs(stacked, vn)

        action, _ = model.predict(normed, deterministic=True)
        obs_flat, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        steps        += 1
        frame_buffer.append(obs_flat.copy())

        # Accumulate path length
        _, _, _, lin_pos, _ = inner.compute_attitude()
        curr_pos     = np.array(lin_pos).flatten()
        path_length += np.linalg.norm(curr_pos - prev_pos)
        prev_pos     = curr_pos.copy()

        # Track nearest obstacle
        obs_dist = inner.state.get("obstacle_distances", None)
        if obs_dist is not None:
            min_obstacle_dist = min(min_obstacle_dist, float(np.min(obs_dist)))

        if terminated or truncated:
            # ---- Determine outcome ----
            if info.get("env_complete", False) or info.get("num_targets_reached", 0) > 0:
                success = True
            elif info.get("tree_collision", False):
                failure_mode = "tree_collision"
            elif info.get("floor_crash", False):
                failure_mode = "floor_crash"
            elif info.get("out_of_bounds", False):
                failure_mode = "out_of_bounds"
            elif info.get("collision", False):
                # base-class collision (plane contact etc.)
                failure_mode = "collision"
            else:
                failure_mode = "timeout"
            break

    if min_obstacle_dist == float("inf"):
        min_obstacle_dist = SENSOR_RANGE   # no trees ever detected

    path_efficiency = path_length / (straight_d + 1e-8)

    return {
        "seed":                  seed,
        "success":               success,
        "failure_mode":          failure_mode,
        "steps":                 steps,
        "path_length":           path_length,
        "straight_line_distance": straight_d,
        "path_efficiency":       path_efficiency,
        "min_obstacle_distance": min_obstacle_dist,
        "total_reward":          total_reward,
    }


# =========================
# MAIN EVALUATION LOOP
# =========================

def run_evaluation(run_id: str, exp_name: str = EXP_NAME):
    base_log_dir = os.path.join(".", "logs", exp_name)
    run_dir      = os.path.join(base_log_dir, run_id)

    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # --- VecNormalize stats -----------------------------------------------
    vecnorm_path = os.path.join(run_dir, "vecnormalize.pkl")
    if not os.path.exists(vecnorm_path):
        raise FileNotFoundError(f"vecnormalize.pkl not found in {run_dir}")

    print(f"Loading VecNormalize stats … {vecnorm_path}")
    vn = load_vecnormalize_stats(vecnorm_path)

    # --- Model ------------------------------------------------------------
    model_path = resolve_model_path(run_dir, run_id)
    print(f"Loading model … {model_path}")
    model = PPO.load(model_path)   # no env needed — we normalise manually
    timestep_label = getattr(model, "num_timesteps", 0)
    print(f"  model.num_timesteps = {timestep_label:,}")

    # --- TensorBoard writer -----------------------------------------------
    timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tb_log_dir = os.path.join(run_dir, "eval", timestamp)
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"TensorBoard log dir: {tb_log_dir}\n")

    # --- Single eval environment ------------------------------------------
    env = make_eval_env()

    # --- Run 100 episodes -------------------------------------------------
    print(f"{'─'*65}")
    print(f"  Running {NUM_TEST_WORLDS} evaluation episodes  (exp={exp_name}, run={run_id})")
    print(f"{'─'*65}")

    results = []
    for i, seed in enumerate(TEST_SEEDS):
        ep = run_episode(env, model, vn, N_STACK, seed)
        results.append(ep)

        if ep["success"]:
            tag = f"✓ steps={ep['steps']:4d}  eff={ep['path_efficiency']:.2f}"
        else:
            tag = f"✗ {ep['failure_mode']}"

        print(f"  [{i+1:3d}/{NUM_TEST_WORLDS}]  seed={seed:3d}  {tag}"
              f"  min_obs={ep['min_obstacle_distance']:.2f}"
              f"  reward={ep['total_reward']:8.1f}")

    env.close()

    # =========================
    # Aggregate metrics
    # =========================
    successes = [r for r in results if r["success"]]
    failures  = [r for r in results if not r["success"]]
    n         = NUM_TEST_WORLDS

    success_rate  = len(successes) / n

    failure_modes = ["tree_collision", "floor_crash", "out_of_bounds", "collision", "timeout"]
    fail_counts   = {m: sum(1 for r in failures if r["failure_mode"] == m) for m in failure_modes}

    mean_reward        = float(np.mean([r["total_reward"]          for r in results]))
    mean_min_obs       = float(np.mean([r["min_obstacle_distance"] for r in results]))
    mean_steps_success = float(np.mean([r["steps"]           for r in successes])) if successes else float("nan")
    mean_path_eff      = float(np.mean([r["path_efficiency"] for r in successes])) if successes else float("nan")

    # =========================
    # TensorBoard — scalars
    # x-axis = model.num_timesteps so multiple eval calls build a learning curve
    # =========================
    writer.add_scalar("eval/success_rate",              success_rate,        timestep_label)
    writer.add_scalar("eval/mean_reward",               mean_reward,         timestep_label)
    writer.add_scalar("eval/mean_min_obstacle_dist",    mean_min_obs,        timestep_label)
    writer.add_scalar("eval/mean_steps_on_success",     mean_steps_success,  timestep_label)
    writer.add_scalar("eval/mean_path_efficiency",      mean_path_eff,       timestep_label)

    for mode in failure_modes:
        writer.add_scalar(f"eval/failure_rate/{mode}", fail_counts[mode] / n, timestep_label)

    # TensorBoard — distributions (gives you per-seed spread, not just mean)
    writer.add_histogram("eval/dist/rewards",
                         np.array([r["total_reward"] for r in results]),
                         timestep_label)
    writer.add_histogram("eval/dist/min_obstacle_distance",
                         np.array([r["min_obstacle_distance"] for r in results]),
                         timestep_label)
    if successes:
        writer.add_histogram("eval/dist/steps_on_success",
                             np.array([r["steps"] for r in successes]),
                             timestep_label)
        writer.add_histogram("eval/dist/path_efficiency",
                             np.array([r["path_efficiency"] for r in successes]),
                             timestep_label)

    writer.close()

    # =========================
    # Print summary
    # =========================
    bar = "=" * 65
    print(f"\n{bar}")
    print(f"  EVALUATION SUMMARY  —  {exp_name} / {run_id}")
    print(f"  (model timesteps: {timestep_label:,})")
    print(bar)
    print(f"  Success rate          : {success_rate*100:5.1f}%   ({len(successes)}/{n})")
    print(f"  Mean reward (all eps) : {mean_reward:8.2f}")
    print(f"  Mean steps (success)  : {mean_steps_success:8.1f}" if successes else
          f"  Mean steps (success)  :      N/A")
    print(f"  Mean path efficiency  : {mean_path_eff:8.3f}   (1.00 = straight line)" if successes else
          f"  Mean path efficiency  :      N/A")
    print(f"  Mean min obstacle     : {mean_min_obs:8.3f}   (higher = safer)")
    print()
    print(f"  Failure breakdown ({len(failures)} failed episodes):")
    for mode in failure_modes:
        c = fail_counts[mode]
        if c:
            print(f"    {mode:<22}: {c:3d}  ({c/n*100:.1f}%)")
    print(bar)
    print(f"  TensorBoard: {tb_log_dir}")
    print(bar)

    return results


# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained QuadX model on a fixed test set.")
    parser.add_argument("--run",  default=RUN_ID,   help=f"Run ID to evaluate (default: {RUN_ID})")
    parser.add_argument("--exp",  default=EXP_NAME, help=f"Experiment name  (default: {EXP_NAME})")
    args = parser.parse_args()

    run_evaluation(run_id=args.run, exp_name=args.exp)