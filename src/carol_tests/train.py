from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from quadx_forest_env import QuadXForestEnv
from PyFlyt.gym_envs import FlattenWaypointEnv

# Create vectorized environment for training
def make_env():
    env = QuadXForestEnv(
        num_trees=5,
        num_targets=1,
        num_sensors=8,
        sensor_range=5.0,
        max_duration_seconds=30.0,
        flight_dome_size=12.0
    )
    env = FlattenWaypointEnv(env, context_length=1)
    return env

env = make_vec_env(make_env, n_envs=4)  # 4 parallel environments

# Train model
model = PPO(
    "MlpPolicy",
    env,
    verbose=0,
    tensorboard_log="./quadx_forest_avoidance_logs/",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99
)

model.learn(total_timesteps=200_000)
model.save("quadx_forest_avoidance")