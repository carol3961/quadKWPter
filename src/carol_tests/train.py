from quadx_forest_env import QuadXForestEnv
from PyFlyt.gym_envs import FlattenWaypointEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

# create vectorized environment for training
def make_env():
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

env = make_vec_env(make_env, n_envs=4)  # 4 parallel environments

model = PPO(
    "MlpPolicy",
    env,
    verbose=0, # 0 for no output, 1 for info, 2 for debug
    tensorboard_log="./quadx_forest_avoidance_logs/",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99
)

model.learn(total_timesteps=100_000, tb_log_name="quadx_waypoints")
model.save("quadx_waypoints")
env.close()

# tensorboard --logdir tb_logs
# type this ^ once ur model is done learning. itll open a local server u can view in ur browser.
# need to pip install tensorflow for this to work
