import numpy as np
from typing import Any, Literal
from gymnasium import spaces
from PyFlyt.gym_envs.quadx_envs.quadx_base_env import QuadXBaseEnv
from PyFlyt.gym_envs.quadx_envs.quadx_waypoints_env import QuadXWaypointsEnv


class QuadXForestEnv(QuadXWaypointsEnv):
    """QuadX Waypoints Environment with trees and obstacle detection"""
    
    def __init__(
        self,
        sparse_reward: bool = False,
        num_targets: int = 1,
        use_yaw_targets: bool = False,
        goal_reach_distance: float = 0.5,
        goal_reach_angle: float = 0.1,
        flight_mode: int = 0,
        flight_dome_size: float = 10.0,
        max_duration_seconds: float = 20.0,
        angle_representation: Literal["euler", "quaternion"] = "quaternion",
        agent_hz: int = 30,
        render_mode: None | Literal["human", "rgb_array"] = None,
        render_resolution: tuple[int, int] = (480, 480),
        # Tree parameters
        num_trees: int = 20,
        tree_radius_range: tuple[float, float] = (0.1, 0.3),
        tree_height_range: tuple[float, float] = (5.0, 5.0),  # CHANGED: Fixed tall height
        tree_collision_penalty: float = 100.0,
        tree_proximity_penalty_weight: float = 0.5,
        goal_area: dict = None,
        # Sensor parameters
        num_sensors: int = 8,
        sensor_range: float = 5.0,
        # NEW: Time penalty
        time_step_penalty: float = 0.1,  # Penalty for each timestep
    ):
        self.tree_positions = []
        self.num_trees = num_trees
        self.tree_radius_range = tree_radius_range
        self.tree_height_range = tree_height_range
        self.tree_collision_penalty = tree_collision_penalty
        self.tree_proximity_penalty_weight = tree_proximity_penalty_weight
        self.time_step_penalty = time_step_penalty
        self.goal_reach_distance = goal_reach_distance
        
        # Sensor configuration
        self.num_sensors = num_sensors
        self.sensor_range = sensor_range
        
        # Default goal area
        if goal_area is None:
            self.goal_area = {
                'x_min': 5.0, 'x_max': 8.0,
                'y_min': 5.0, 'y_max': 8.0,
                'z_min': 1.5, 'z_max': 2.5,
            }
        else:
            self.goal_area = goal_area
        
        # Call parent init
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
        
        # Override observation space to include sensors
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
    ) -> tuple[dict[Literal["attitude", "target_deltas", "obstacle_distances"], np.ndarray], dict]:
        """Resets the environment with trees and custom spawn positions"""

        QuadXBaseEnv.begin_reset(self, seed, options)
        
        # Parent's waypoint reset
        self.waypoints.reset(self.env, self.np_random)
        
        # Generate goal in constrained area
        goal_x = self.np_random.uniform(self.goal_area['x_min'], 
                                        self.goal_area['x_max'])
        goal_y = self.np_random.uniform(self.goal_area['y_min'], 
                                        self.goal_area['y_max'])
        goal_z = self.np_random.uniform(self.goal_area['z_min'], 
                                        self.goal_area['z_max'])
        
        desired_goal = np.array([goal_x, goal_y, goal_z])
        
        # Update logical and visual target position
        self.waypoints.targets[0] = desired_goal
        
        if self.waypoints.enable_render and len(self.waypoints.target_visual) > 0:
            self.env.resetBasePositionAndOrientation(
                self.waypoints.target_visual[0],
                desired_goal.tolist(),
                [0, 0, 0, 1]
            )
        
        # Generate trees
        self._generate_trees()
        
        # Reset distance tracking
        self.previous_distance = np.linalg.norm(self.start_pos[0] - desired_goal)
        
        self.info["num_targets_reached"] = 0
        QuadXBaseEnv.end_reset(self)
        
        return self.state, self.info
    
    def compute_state(self) -> None:
        """Computes the state including obstacle sensor readings"""
        
        # Get base state from parent
        super().compute_state()
        
        # Add obstacle sensor readings
        obstacle_distances = self._get_obstacle_distances()
        self.state["obstacle_distances"] = obstacle_distances
    
    def _get_obstacle_distances(self) -> np.ndarray:
        """Cast rays around the drone to detect obstacles"""
        
        # Get drone state
        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = self.compute_attitude()
        
        # Get rotation matrix (quaternion is a tuple, convert to list)
        rotation_matrix = np.array(
            self.env.getMatrixFromQuaternion(list(quaternion))
        ).reshape(3, 3)
        
        # Get drone position as numpy array
        drone_pos = np.array(lin_pos).flatten()
        
        # Initialize distances to max range
        distances = np.full(self.num_sensors, self.sensor_range, dtype=np.float64)
        
        # Cast rays around drone
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
    
    def compute_state_reward(self):
        # --- Progress toward goal ---
        current_dist = np.linalg.norm(
            self.env.state(0)[-1][:3] - self.targets[self.current_target_index]
        )
        progress = self.previous_distance - current_dist
        self.previous_distance = current_dist
        # progress_reward = 5.0 * progress
        progress_reward = 7.0 * progress

        # --- Survival (encourages staying alive → encourages dodging) ---
        # survival_reward = 0.5
        survival_reward = -0.5

        # --- Obstacle proximity (exponential) ---
        obstacle_penalty = 0.0
        danger_radius = 3.0
        if hasattr(self, "obstacle_distances"):
            min_dist = np.min(self.obstacle_distances)
            if min_dist < danger_radius:
                normalized = min_dist / danger_radius
                obstacle_penalty = -3.0 * (1.0 - normalized) ** 2
                # At distance 0.5 → penalty = -1.9
                # At distance 1.0 → penalty = -1.3
                # At distance 2.0 → penalty = -0.3

        # --- Goal reached bonus ---
        goal_bonus = 0.0
        goal_threshold = 1.0 
        if current_dist < goal_threshold:
            goal_bonus = 50.0
        total += goal_bonus

        # --- Velocity toward goal (rewards speed in the right direction) ---
        velocity = self.env.state(0)[-1][3:6]  # adjust based on your state layout
        goal_direction = (
            self.targets[self.current_target_index] - self.env.state(0)[-1][:3]
        )
        goal_direction_norm = goal_direction / (np.linalg.norm(goal_direction) + 1e-8)
        speed_toward_goal = np.dot(velocity, goal_direction_norm)
        velocity_reward = 2.0 * max(speed_toward_goal, 0.0)


        # --- Collision (devastating) ---
        collision_penalty = 0.0
        if self.env.contact_array[0]:  # however you detect tree collision
            collision_penalty = -100.0

        total = progress_reward + survival_reward + obstacle_penalty + collision_penalty
        return total
    
    def _check_tree_collision(self) -> bool:
        """Check if drone has collided with any tree"""
        
        # Get drone's body ID from the aviary
        try:
            drone_id = self.env.drones[0].Id
        except:
            # Alternative method
            return False
        
        # Get all contact points for the drone
        contact_points = self.env.getContactPoints(bodyA=drone_id)
        
        # Check if any contact is with a tree
        for contact in contact_points:
            body_b = contact[2]
            
            for tree in self.tree_positions:
                if body_b == tree['id']:
                    return True
        
        return False
    
    def _generate_trees(self):
        """Generates trees between drone start and waypoint"""
        
        self.tree_positions = []
        
        # Get start and goal positions
        start = self.start_pos[0][:2]
        goal = self.waypoints.targets[0][:2]
        
        for _ in range(self.num_trees):
            is_valid_position = False
            attempts = 0
            
            while not is_valid_position and attempts < 100:
                # Sample along the path
                t = self.np_random.uniform(0.0, 1.0)
                center_point = start + t * (goal - start)
                
                # Perpendicular offset
                corridor_width = 2.0
                direction = goal - start
                direction_norm = direction / (np.linalg.norm(direction) + 1e-8)
                perpendicular = np.array([-direction_norm[1], direction_norm[0]])
                offset_distance = self.np_random.uniform(-corridor_width, corridor_width)
                
                xy_pos = center_point + offset_distance * perpendicular
                x, y = xy_pos
                z = 0
                pos = np.array([x, y, z])
                
                # Validation checks
                if np.linalg.norm(pos[:2] - self.start_pos[0][:2]) < 0.8:
                    attempts += 1
                    continue
                
                is_too_close = False
                if hasattr(self.waypoints, 'targets') and len(self.waypoints.targets) > 0:
                    waypoint = self.waypoints.targets[0]
                    if np.linalg.norm(pos[:2] - waypoint[:2]) < 1.0:
                        is_too_close = True
                
                if np.linalg.norm(pos[:2]) > self.flight_dome_size / 2:
                    attempts += 1
                    continue
                
                if not is_too_close:
                    is_valid_position = True
                    # CHANGED: Use max height from range (or fixed tall height)
                    height = self.tree_height_range[1]  # Use max height
                    radius = self.np_random.uniform(*self.tree_radius_range)
                    
                    collision_shape = self.env.createCollisionShape(
                        shapeType=self.env.GEOM_CYLINDER,
                        radius=radius,
                        height=height
                    )
                    
                    visual_shape = self.env.createVisualShape(
                        shapeType=self.env.GEOM_CYLINDER,
                        radius=radius,
                        length=height,
                        rgbaColor=[0.55, 0.27, 0.07, 1],
                        specularColor=[0.4, 0.4, 0]
                    )
                    
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