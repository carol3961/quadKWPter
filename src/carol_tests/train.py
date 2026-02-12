import gymnasium as gym
from PyFlyt.gym_envs import FlattenWaypointEnv
from stable_baselines3 import PPO
from quadx_forest_env import QuadXForestEnv

# adjust training timestep as needed. 500,000 takes my computer around 10 minutes fyi

#env = gym.make("PyFlyt/QuadX-Waypoints-v4")
env = QuadXForestEnv(
    render_mode=None,
    num_trees=10,
    tree_radius_range=(0.1, 0.3),
    tree_height_range=(2.0, 4.0),
    num_targets=1
)
env = FlattenWaypointEnv(env, context_length=2)

model = PPO(
    "MlpPolicy",
    env,
    verbose=0,  # 0 for no output, 1 for info, 2 for debug
    tensorboard_log="./tb_logs", 
    n_steps=2048,
    batch_size=64,
    learning_rate=3e-4,
)

model.learn(total_timesteps=100_000, tb_log_name="quadx_waypoints")
model.save("quadx_waypoints")
env.close()

# tensorboard --logdir tb_logs
# type this ^ once ur model is done learning. itll open a local server u can view in ur browser. 
# need to pip install tensorflow for this to work
