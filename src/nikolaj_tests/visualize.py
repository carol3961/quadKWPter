import os
import numpy as np
import imageio.v2 as imageio
import imageio_ffmpeg
from datetime import datetime


from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, VecFrameStack

from quadx_forest_env import QuadXForestEnv
from PyFlyt.gym_envs import FlattenWaypointEnv

os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()

MODEL_PATH = "checkpoints3/v4_save_params/quadx_checkpoints_500000_steps.zip"
NORM_PATH  = "checkpoints3/v4_save_params/quadx_checkpoints_vecnormalize_500000_steps.pkl"

FPS = 30
MAX_STEPS = 1500
RENDER_EVERY = 4  # <-- important

def make_env():
    env = QuadXForestEnv(
        num_trees=5,
        num_targets=1,
        num_sensors=8,
        sensor_range=5.0,
        max_duration_seconds=30.0,
        flight_dome_size=12.0,
        render_mode="rgb_array",
        render_resolution=(256, 256),  # <-- lower res = faster
    )
    env = FlattenWaypointEnv(env, context_length=2)
    return env

env = DummyVecEnv([make_env])
env = VecFrameStack(env, n_stack=4)

env = VecNormalize.load(NORM_PATH, env)
env.training = False
env.norm_reward = False

model = PPO.load(MODEL_PATH, env=env)

obs = env.reset()
frames = []

for t in range(MAX_STEPS):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    if t % RENDER_EVERY == 0:
        frame = env.venv.envs[0].render()  # base env render
        if frame is not None:
            frames.append(np.asarray(frame))

    if done[0]:
        obs = env.reset()

env.close()

# If we rendered every 4 steps, video FPS should be scaled down accordingly
video_fps = FPS // RENDER_EVERY if FPS // RENDER_EVERY > 0 else 1
run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
vid_name = f"quadKWPter_logs/{run_id}.mp4"
imageio.mimwrite(vid_name, frames, fps=video_fps, quality=8)
print("Wrote rollout.mp4", "frames:", len(frames), "fps:", video_fps)
