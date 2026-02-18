import numpy as np
import os
import pybullet as p
from gymnasium import spaces
from typing import Any, Literal
from PyFlyt.gym_envs.quadx_envs.quadx_base_env import QuadXBaseEnv
from PyFlyt.gym_envs.quadx_envs.quadx_waypoints_env import QuadXWaypointsEnv


class QuadXForestEnv(QuadXWaypointsEnv):
    """QuadX Waypoints Environment with trees"""
    
    def __init__(
        self,
        sparse_reward: bool = False,
        num_targets: int = 4,
        use_yaw_targets: bool = False,
        goal_reach_distance: float = 0.2,
        goal_reach_angle: float = 0.1,
        flight_mode: int = 0,
        flight_dome_size: float = 5.0,
        max_duration_seconds: float = 10.0,
        angle_representation: Literal["euler", "quaternion"] = "quaternion",
        agent_hz: int = 30,
        render_mode: None | Literal["human", "rgb_array"] = None,
        render_resolution: tuple[int, int] = (480, 480),
        # default parameters to model trees
        num_trees: int = 10,
        tree_radius_range: tuple[float, float] = (0.1, 0.2),
        tree_height_range: tuple[float, float] = (0.4, 0.8),
        tree_mesh_dir_path: str = os.path.join(os.getcwd(), "gazebo_pine_tree_model", "meshes"),

        # default parameters to train drone in tree env
        tree_collision_penalty: float = 100.0,  # tree penalties
        tree_proximity_penalty: float = 0.5,
        num_sensors: int = 8,                   # sensor configs
        sensor_range: float = 5.0,
        time_step_penalty: float = 0.1,         # time penalty
        goal_area: dict = None,                 # goal area
    ):
        
        # added attributes to model trees 
        self.tree_positions = []
        self.num_trees = num_trees
        self.tree_radius_range = tree_radius_range
        self.tree_height_range = tree_height_range
        self.tree_mesh_dir_path = tree_mesh_dir_path
        self.tree_mesh_path = self._get_tree_mesh_path()

        # added attributes to train drone in tree env 
        self.tree_collision_penalty = tree_collision_penalty
        self.tree_proximity_penalty = tree_proximity_penalty
        self.num_sensors = num_sensors
        self.sensor_range = sensor_range
        self.time_step_penalty = time_step_penalty  
        if goal_area is None:
            self.goal_area = {
                'x_min': 5.0, 'x_max': 8.0,
                'y_min': 5.0, 'y_max': 8.0,
                'z_min': 1.5, 'z_max': 2.5,
            }
        else:
            self.goal_area = goal_area
        
        # call parent (QuadXWaypointsEnv) init
        super().__init__(
            sparse_reward=sparse_reward,
            num_targets=num_targets,
            use_yaw_targets=use_yaw_targets,
            goal_reach_distance=goal_reach_distance,
            goal_reach_angle=goal_reach_angle,
            flight_mode=flight_mode,
            flight_dome_size=flight_dome_size,
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,
            agent_hz=agent_hz,
            render_mode=render_mode,
            render_resolution=render_resolution,
        )

        # override observation space to include sensors
        self.observation_space = spaces.Dict(
            {
                "attitude": self.combined_space,
                "target_deltas": spaces.Sequence(
                    space=spaces.Box(
                        low=-2 * flight_dome_size,
                        high=2 * flight_dome_size,
                        shape=(4,) if use_yaw_targets else (3,),
                        dtype=np.float64,
                    ),
                    stack=True,
                ),
                "obstacle_distances": spaces.Box(
                    low=0.0,
                    high=sensor_range,
                    shape=(num_sensors,),
                    dtype=np.float64,
                ),
            }
        )
        

    def reset(
        self, *, seed: None | int = None, options: None | dict[str, Any] = dict()
    ) -> tuple[dict[Literal["attitude", "target_deltas"], np.ndarray], dict]:
        """Resets the environment with trees"""
        QuadXBaseEnv.begin_reset(self, seed, options)
        self.waypoints.reset(self.env, self.np_random)

        # generate goal in constrained area
        goal_x = self.np_random.uniform(self.goal_area['x_min'], self.goal_area['x_max'])
        goal_y = self.np_random.uniform(self.goal_area['y_min'], self.goal_area['y_max'])
        goal_z = self.np_random.uniform(self.goal_area['z_min'], self.goal_area['z_max'])
        desired_goal = np.array([goal_x, goal_y, goal_z])
        
        # update logical and visual target position
        self.waypoints.targets[0] = desired_goal
        if self.waypoints.enable_render and len(self.waypoints.target_visual) > 0:
            self.env.resetBasePositionAndOrientation(
                self.waypoints.target_visual[0],
                desired_goal.tolist(),
                [0, 0, 0, 1]
            )
        
        # generate trees
        self._generate_trees()

        # reset distance tracking
        self.previous_distance = np.linalg.norm(self.start_pos[0] - desired_goal)
        self.info["num_targets_reached"] = 0
        QuadXBaseEnv.end_reset(self)
        
        return self.state, self.info
    

    def compute_state(self) -> None:
        """Computes the state including obstacle sensor readings"""
        # get base state from parent
        super().compute_state()
        # add obstacle sensor readings
        obstacle_distances = self._get_obstacle_distances()
        self.state["obstacle_distances"] = obstacle_distances
    

    def _get_obstacle_distances(self) -> np.ndarray:
        """Cast rays around the drone to detect obstacles"""
        # get drone state
        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = self.compute_attitude()
        # get rotation matrix (quaternion is a tuple, convert to list)
        rotation_matrix = np.array(
            self.env.getMatrixFromQuaternion(list(quaternion))
        ).reshape(3, 3)
        
        # get drone position as numpy array
        drone_pos = np.array(lin_pos).flatten()
        
        # initialize distances to max range
        distances = np.full(self.num_sensors, self.sensor_range, dtype=np.float64)
        
        # cast rays around drone
        for i in range(self.num_sensors):
            angle = 2 * np.pi * i / self.num_sensors
            
            local_dir = np.array([np.cos(angle), np.sin(angle), 0.0])
            world_dir = rotation_matrix @ local_dir
            
            ray_from = drone_pos
            ray_to = ray_from + world_dir * self.sensor_range
            
            ray_result = self.env.rayTest(ray_from.tolist(), ray_to.tolist())
            
            if len(ray_result) > 0 and ray_result[0][0] >= 0:
                distances[i] = ray_result[0][2] * self.sensor_range
        
        return distances
    

    def compute_term_trunc_reward(self) -> None:
        """Compute reward with stronger goal-seeking incentive"""
        # call parent for base termination/reward
        super().compute_term_trunc_reward()
        
        # check if targets still exist (might be empty after reaching goal)
        if len(self.waypoints.targets) == 0:
            return  # episode is ending, skip distance calculations
        
        # get current distance to goal
        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = self.compute_attitude()
        goal_pos = self.waypoints.targets[0]
        current_distance = np.linalg.norm(lin_pos - goal_pos)
        
        # track previous distance for progress reward
        if not hasattr(self, 'previous_distance'):
            self.previous_distance = current_distance
        
        # strong progress reward/penalty
        progress = self.previous_distance - current_distance
        #self.reward += 20.0 * progress
        self.reward += 5.0 * np.clip(progress, -0.5, 0.5)
        self.previous_distance = current_distance
        
        # strong proximity reward
        self.reward += 1.0 / (current_distance + 0.1)
        
        # time penalty
        self.reward -= self.time_step_penalty
        
        # penalty for going too high
        goal_height = goal_pos[2]
        current_height = lin_pos[2]
        if current_height > goal_height + 2.0:
            height_penalty = (current_height - goal_height - 2.0) * 0.5
            self.reward -= height_penalty
        
        # tree collision
        if self._check_tree_collision():
            self.reward = -self.tree_collision_penalty
            self.termination = True
            self.info["tree_collision"] = True
            return
        
        # proximity penalty
        if not self.sparse_reward:
            obstacle_distances = self.state.get("obstacle_distances", None)
            if obstacle_distances is not None:
                min_distance = np.min(obstacle_distances)
                if min_distance < 1.5:
                    proximity_penalty = self.tree_proximity_penalty * (1.5 - min_distance)
                    self.reward -= proximity_penalty

    
    def _check_tree_collision(self) -> bool:
        """Check if drone has collided with any tree"""
        # get drone's body ID from the aviary
        try:
            drone_id = self.env.drones[0].Id
        except:
            # alternative method
            return False
        
        # get all contact points for the drone
        contact_points = self.env.getContactPoints(bodyA=drone_id)
        
        # check if any contact is with a tree
        for contact in contact_points:
            body_b = contact[2]
            
            for tree in self.tree_positions:
                if body_b == tree['id']:
                    return True
        
        return False


    def _get_tree_mesh_path(self):
        """Finds the mesh file for the pine tree model"""
        if os.path.exists(self.tree_mesh_dir_path):
            mesh_files = [f for f in os.listdir(self.tree_mesh_dir_path) if f.endswith((".dae", ".obj", ".stl"))]
            if mesh_files:
                mesh_path = os.path.join(self.tree_mesh_dir_path, mesh_files[0])
                return mesh_path
        raise FileNotFoundError(f"No mesh files found in {self.tree_mesh_dir_path}")
    

    def _generate_trees(self):
        """Randomly generates trees in the environment, avoiding points that contain 
        waypoints and the starting position"""
        self.tree_positions = []
        for _ in range(self.num_trees):
            attempts = 0
            is_valid_position = False
            
            while attempts < 20:
                # randomly position a tree within the flight dome at point (x, y, z (ground))
                x = self.np_random.uniform(-self.flight_dome_size, self.flight_dome_size)
                y = self.np_random.uniform(-self.flight_dome_size, self.flight_dome_size)
                z = 0  
                pos = np.array([x, y, z])
                
                # check tree is far enough from drone starting position 
                if np.linalg.norm(pos[:2] - self.start_pos[0][:2]) < 2:
                    attempts += 1
                    continue
                
                # check tree is far enough from waypoints 
                is_too_close = False
                if hasattr(self.waypoints, "targets"):
                    for waypoint in self.waypoints.targets:
                        if np.linalg.norm(pos[:2] - waypoint[:2]) < 2:
                            is_too_close = True
                            break
                if is_too_close:
                    attempts += 1
                    continue
                
                # passed checks, is valid position
                is_valid_position = True
                break

            if not is_valid_position:
                continue

            # random tree size and orientation
            height = self.np_random.uniform(*self.tree_height_range)
            radius = self.np_random.uniform(*self.tree_radius_range)
            rotation = self.np_random.uniform(0, 2*np.pi)
            orientation = p.getQuaternionFromEuler([0, 0, rotation])
            
            # load Gazebo tree mesh
            visual_shape = self.env.createVisualShape(
                shapeType=self.env.GEOM_MESH,
                fileName=self.tree_mesh_path,
                meshScale=[height, height, height],
                rgbaColor=[178/255, 172/255, 136/255, 1]
            )
        
            # use cylinder collision shape
            collision_shape = self.env.createCollisionShape(
                shapeType=self.env.GEOM_CYLINDER,
                radius=radius,
                height=height
            )
            
            # create tree
            tree_id = self.env.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[x, y, z],
                baseOrientation=orientation
            )
            self.tree_positions.append({
                'id': tree_id,
                'position': np.array([x, y, z]),
                'radius': radius,
                'height': height
            })    
            
        attempts += 1
