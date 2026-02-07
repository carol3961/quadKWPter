import gymnasium as gym
import PyFlyt.gym_envs
from PyFlyt.gym_envs import FlattenWaypointEnv
from stable_baselines3 import PPO
from tensorboard_video_recorder import TensorboardVideoRecorder
from datetime import datetime  # Added this
from stable_baselines3.common.monitor import Monitor


# adjust training timestep as needed. 500,000 takes my computer around 10 minutes fyi
experiment_name = "cluster_run"
experiment_logdir = f"quadKWPter_logs/{experiment_name}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

env = gym.make("PyFlyt/QuadX-Waypoints-v4", render_mode="rgb_array")
env = Monitor(env)
env = FlattenWaypointEnv(env, context_length=2)

video_trigger = lambda step: step % 2000 == 0    # Wrap the environment in a monitor that records videos to Tensorboard
env = TensorboardVideoRecorder(env=env,
                                   video_trigger=video_trigger,
                                   video_length=2000,
                                   fps=30,
                                   record_video_env_idx=0,
                                   tb_log_dir=experiment_logdir)


model = PPO(
    "MlpPolicy",
    env,
    verbose=0,  # 0 for no output, 1 for info, 2 for debug
    tensorboard_log=experiment_logdir,
    n_steps=2048,
    batch_size=64,
    learning_rate=3e-4,
)


model.learn(total_timesteps=1_000_000, tb_log_name="quadx_waypoints")
model.save("quadx_waypoints")
env.close()

# tensorboard --bind_all --port 8887 --logdir quadKWPter_logs
# type this ^ once ur model is done learning. itll open a local server u can view in ur browser.
# need to pip install tensorflow for this to work
