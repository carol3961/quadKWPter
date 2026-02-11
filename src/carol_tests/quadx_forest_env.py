import numpy as np
from typing import Any, Literal
from gymnasium import spaces
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
        num_trees: int = 20,
        tree_radius_range: tuple[float, float] = (0.1, 0.3),
        tree_height_range: tuple[float, float] = (2.0, 4.0),
        tree_collision_penalty: float = 100.0,
        tree_proximity_penalty_weight: float = 0.5,
    ):
        # <--- added attributes to model trees --->
        self.tree_positions = []
        self.num_trees = num_trees
        self.tree_radius_range = tree_radius_range
        self.tree_height_range = tree_height_range
        self.tree_collision_penalty = tree_collision_penalty
        self.tree_proximity_penalty_weight = tree_proximity_penalty_weight
        
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
        # <--- randomly generate trees --->
        self._generate_trees()
        self.info["num_targets_reached"] = 0
        QuadXBaseEnv.end_reset(self)
        
        return self.state, self.info
    

    def _generate_trees(self):
        """Randomly generates trees in the environment, avoiding points that contain 
        waypoints and the starting position"""

        for _ in range(self.num_trees):
            is_valid_position = False
            attempts = 0
            
            while not is_valid_position and attempts < 100:
                # randomly position a tree within the flight dome at point (x, y, z (ground))
                x = self.np_random.uniform(-self.flight_dome_size/2, self.flight_dome_size/2)
                y = self.np_random.uniform(-self.flight_dome_size/2, self.flight_dome_size/2)
                z = 0  
                pos = np.array([x, y, z])
                
                # check tree is far enough from drone starting position 
                if np.linalg.norm(pos[:2] - self.start_pos[0][:2]) < 1.5:
                    attempts += 1
                    continue
                
                # check tree is far enough from waypoints 
                is_too_close = False
                if hasattr(self.waypoints, 'targets'):
                    for waypoint in self.waypoints.targets:
                        if np.linalg.norm(pos[:2] - waypoint[:2]) < 1.5:
                            is_too_close = True
                            break
                
                if not is_too_close:
                    is_valid_position = True
                    height = self.np_random.uniform(*self.tree_height_range)
                    radius = self.np_random.uniform(*self.tree_radius_range)
                    
                    # create cylinder collision shape to model trees
                    collision_shape = self.env.createCollisionShape(
                        shapeType=self.env.GEOM_CYLINDER,
                        radius=radius,
                        height=height
                    )
                    
                    # create cylinder visual shape
                    visual_shape = self.env.createVisualShape(
                        shapeType=self.env.GEOM_CYLINDER,
                        radius=radius,
                        length=height,
                        rgbaColor=[0.55, 0.27, 0.07, 1],
                        specularColor=[0.4, 0.4, 0]
                    )
                    
                    # create the tree body
                    tree_id = self.env.createMultiBody(
                        baseMass=0,
                        baseCollisionShapeIndex=collision_shape,
                        baseVisualShapeIndex=visual_shape,
                        basePosition=[x, y, height/2],
                        baseOrientation=[0, 0, 0, 1]
                    )
                    
                    self.tree_positions.append({
                        'id': tree_id,
                        'position': np.array([x, y, height/2]),
                        'radius': radius,
                        'height': height
                    })
                
                attempts += 1

