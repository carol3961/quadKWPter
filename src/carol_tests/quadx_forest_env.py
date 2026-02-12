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
        goal_area: dict = None,
    ):
        # <--- added attributes to model trees --->
        self.tree_positions = []
        self.num_trees = num_trees
        self.tree_radius_range = tree_radius_range
        self.tree_height_range = tree_height_range
        self.tree_collision_penalty = tree_collision_penalty
        self.tree_proximity_penalty_weight = tree_proximity_penalty_weight

        # default goal area
        if goal_area is None:
            self.goal_area = {
                'x_min': 3.0, 'x_max': 3.0,
                'y_min': 3.0, 'y_max': 3.0,
                'z_min': 1, 'z_max': 2,
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
        
    def reset(
        self, *, seed: None | int = None, options: None | dict[str, Any] = dict()
    ) -> tuple[dict[Literal["attitude", "target_deltas"], np.ndarray], dict]:
        """Resets the environment with trees and custom spawn positions"""

        QuadXBaseEnv.begin_reset(self, seed, options)
        
        # Parent's waypoint reset (creates visual at random position)
        self.waypoints.reset(self.env, self.np_random)
        
        # Generate goal in constrained area
        goal_x = self.np_random.uniform(self.goal_area['x_min'], 
                                        self.goal_area['x_max'])
        goal_y = self.np_random.uniform(self.goal_area['y_min'], 
                                        self.goal_area['y_max'])
        goal_z = self.np_random.uniform(self.goal_area['z_min'], 
                                        self.goal_area['z_max'])
        
        desired_goal = np.array([goal_x, goal_y, goal_z])
        
        # Update logical target position
        self.waypoints.targets[0] = desired_goal
        
        # UPDATE VISUAL TARGET POSITION
        if self.waypoints.enable_render and len(self.waypoints.target_visual) > 0:
            self.env.resetBasePositionAndOrientation(
                self.waypoints.target_visual[0],
                desired_goal.tolist(),
                [0, 0, 0, 1]
            )
        
        # Generate trees (now avoiding the CORRECT goal position)
        self._generate_trees()
        
        self.info["num_targets_reached"] = 0
        QuadXBaseEnv.end_reset(self)
        
        return self.state, self.info
    
    def _generate_trees(self):
        """Generates trees between drone start and waypoint"""
        
        self.tree_positions = []  # Clear previous trees
        
        # Get start and goal positions
        start = self.start_pos[0][:2]  # [x, y] only (ignore z)
        goal = self.waypoints.targets[0][:2]  # [x, y] of waypoint
        
        for _ in range(self.num_trees):
            is_valid_position = False
            attempts = 0
            
            while not is_valid_position and attempts < 100:
                # <--- Sample position BETWEEN start and goal --->
                
                # Method 1: Sample along the path with random offset
                # t = 0 is start, t = 1 is goal
                t = self.np_random.uniform(0.0, 1.0)  # Progress along path
                
                # Interpolate between start and goal
                center_point = start + t * (goal - start)
                
                # Add perpendicular offset (width of corridor)
                corridor_width = 2.0  # Width of tree corridor in meters
                
                # Direction vector and perpendicular
                direction = goal - start
                direction_norm = direction / (np.linalg.norm(direction) + 1e-8)
                perpendicular = np.array([-direction_norm[1], direction_norm[0]])
                
                # Random offset perpendicular to path
                offset_distance = self.np_random.uniform(-corridor_width, corridor_width)
                
                # Final position
                xy_pos = center_point + offset_distance * perpendicular
                x, y = xy_pos
                z = 0
                pos = np.array([x, y, z])
                
                # <--- Validation checks --->
                
                # Check distance from start position (but allow closer since we want obstacles)
                if np.linalg.norm(pos[:2] - self.start_pos[0][:2]) < 0.8:
                    attempts += 1
                    continue
                
                # Check distance from waypoint
                is_too_close = False
                if hasattr(self.waypoints, 'targets') and len(self.waypoints.targets) > 0:
                    waypoint = self.waypoints.targets[0]
                    if np.linalg.norm(pos[:2] - waypoint[:2]) < 1.0:
                        is_too_close = True
                
                # Check it's within flight dome
                if np.linalg.norm(pos[:2]) > self.flight_dome_size / 2:
                    attempts += 1
                    continue
                
                if not is_too_close:
                    is_valid_position = True
                    height = self.np_random.uniform(*self.tree_height_range)
                    radius = self.np_random.uniform(*self.tree_radius_range)
                    
                    # Create cylinder collision shape
                    collision_shape = self.env.createCollisionShape(
                        shapeType=self.env.GEOM_CYLINDER,
                        radius=radius,
                        height=height
                    )
                    
                    # Create cylinder visual
                    visual_shape = self.env.createVisualShape(
                        shapeType=self.env.GEOM_CYLINDER,
                        radius=radius,
                        length=height,
                        rgbaColor=[0.55, 0.27, 0.07, 1],
                        specularColor=[0.4, 0.4, 0]
                    )
                    
                    # Create tree body
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

