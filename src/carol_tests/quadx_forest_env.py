import numpy as np
import os
import pybullet as p
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
        # <--- default parameters for trees --->
        num_trees: int = 5,
        tree_radius_range: tuple[float, float] = (0.1, 0.2),
        tree_height_range: tuple[float, float] = (0.4, 0.8),
        tree_collision_penalty: float = 100.0,
        tree_proximity_penalty_weight: float = 0.5,
        tree_mesh_dir_path: str = os.path.join(os.getcwd(), "gazebo_models", "pine_tree", "meshes"),
    ):
        # <--- added attributes to model trees --->
        self.tree_positions = []
        self.num_trees = num_trees
        self.tree_radius_range = tree_radius_range
        self.tree_height_range = tree_height_range
        self.tree_collision_penalty = tree_collision_penalty
        self.tree_proximity_penalty_weight = tree_proximity_penalty_weight
        self.tree_mesh_dir_path = tree_mesh_dir_path
        self.tree_mesh_path = self._get_tree_mesh_path()
        
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
        

    def reset(
        self, *, seed: None | int = None, options: None | dict[str, Any] = dict()
    ) -> tuple[dict[Literal["attitude", "target_deltas"], np.ndarray], dict]:
        """Resets the environment with trees"""
        QuadXBaseEnv.begin_reset(self, seed, options)
        self.waypoints.reset(self.env, self.np_random)
        # <--- generate trees --->
        self._generate_trees()
        self.info["num_targets_reached"] = 0
        QuadXBaseEnv.end_reset(self)
        
        return self.state, self.info
    

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
            
            while attempts < 50:
                # randomly position a tree within the flight dome at point (x, y, z (ground))
                x = self.np_random.uniform(-self.flight_dome_size/2, self.flight_dome_size/2)
                y = self.np_random.uniform(-self.flight_dome_size/2, self.flight_dome_size/2)
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
            
            