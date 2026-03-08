import numpy as np
from typing import Any, Literal
from gymnasium import spaces
from PyFlyt.gym_envs.quadx_envs.quadx_base_env import QuadXBaseEnv
from PyFlyt.gym_envs.quadx_envs.quadx_waypoints_env import QuadXWaypointsEnv
import os
import pybullet as p


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
        tree_height_range: tuple[float, float] = (1.0, 2.0), 
        tree_mesh_dir_path: str = os.path.join(os.getcwd(), "gazebo_pine_tree_model", "meshes"),
        tree_collision_penalty: float = 100.0,
        tree_proximity_penalty_weight: float = 0.5,  
        goal_area: dict = None,
        # Sensor parameters
        num_sensors: int = 8,
        sensor_range: float = 5.0,
        # Time penalty
        time_step_penalty: float = 0.1,  
    ):
        self.tree_positions = []
        self.num_trees = num_trees
        self.tree_radius_range = tree_radius_range
        self.tree_height_range = tree_height_range
        self.tree_mesh_dir_path = tree_mesh_dir_path
        self.tree_mesh_path = self._get_tree_mesh_path()
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
        goal_x = self.np_random.uniform(self.goal_area['x_min'], self.goal_area['x_max'])
        goal_y = self.np_random.uniform(self.goal_area['y_min'], self.goal_area['y_max'])
        goal_z = self.np_random.uniform(self.goal_area['z_min'], self.goal_area['z_max'])
        
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
        
        # Get rotation matrix
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
            
    def _check_tree_collision(self) -> bool:
        """Check if drone has collided with any tree"""
        try:
            drone_id = self.env.drones[0].Id
        except:
            return False
        
        contact_points = self.env.getContactPoints(bodyA=drone_id)
        
        for contact in contact_points:
            body_b = contact[2]
            for tree in self.tree_positions:
                if body_b == tree['id']:
                    return True
        
        return False
    
    def _get_tree_mesh_path(self):
        """Finds the mesh file for the pine tree model"""
        if os.path.exists(self.tree_mesh_dir_path):
            mesh_files = [f for f in os.listdir(self.tree_mesh_dir_path) 
                         if f.endswith((".dae", ".obj", ".stl"))]
            if mesh_files:
                mesh_path = os.path.join(self.tree_mesh_dir_path, mesh_files[0])
                return mesh_path
        raise FileNotFoundError(f"No mesh files found in {self.tree_mesh_dir_path}")
    
    def _generate_trees(self):
        """Randomly generates trees avoiding waypoints and starting position"""
        self.tree_positions = []
        
        for _ in range(self.num_trees):
            attempts = 0
            is_valid_position = False
            
            while attempts < 50:
                x = self.np_random.uniform(-self.flight_dome_size, self.flight_dome_size)
                y = self.np_random.uniform(-self.flight_dome_size, self.flight_dome_size)
                z = 0  
                pos = np.array([x, y, z])
                
                # Check distance from start
                if np.linalg.norm(pos[:2] - self.start_pos[0][:2]) < 2:  
                    attempts += 1
                    continue
                
                # Check distance from waypoints
                is_too_close = False
                if hasattr(self.waypoints, "targets"):
                    for waypoint in self.waypoints.targets:
                        if np.linalg.norm(pos[:2] - waypoint[:2]) < 2:  
                            is_too_close = True
                            break
                
                if is_too_close:
                    attempts += 1
                    continue
                
                is_valid_position = True
                break

            if not is_valid_position:
                continue

            height = self.np_random.uniform(*self.tree_height_range)
            radius = self.np_random.uniform(*self.tree_radius_range)
            rotation = self.np_random.uniform(0, 2*np.pi)
            orientation = p.getQuaternionFromEuler([0, 0, rotation])
            
            visual_shape = self.env.createVisualShape(
                shapeType=self.env.GEOM_MESH,
                fileName=self.tree_mesh_path,
                meshScale=[height, height, height],
                rgbaColor=[178/255, 172/255, 136/255, 1]
            )
        
            collision_shape = self.env.createCollisionShape(
                shapeType=self.env.GEOM_CYLINDER,
                radius=radius,
                height=height
            )
            
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
    
    def render(self):
        """Third-person chase camera render"""
        if getattr(self, "render_mode", None) != "rgb_array":
            try:
                return super().render()
            except Exception:
                return None

        width, height = getattr(self, "render_resolution", (480, 480))
        _, _, _, lin_pos, _ = self.compute_attitude()
        target = np.array(lin_pos).flatten().tolist()

        distance = 6.0
        yaw = 45
        pitch = -30
        roll = 0

        view_matrix = self.env.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=target,
            distance=distance,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            upAxisIndex=2,
        )

        proj_matrix = self.env.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(width) / float(height),
            nearVal=0.1,
            farVal=100.0,
        )

        _, _, rgba, _, _ = self.env.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=getattr(self.env, "ER_BULLET_HARDWARE_OPENGL", 0),
        )

        rgba = np.asarray(rgba)
        if rgba.ndim == 1:
            rgba = rgba.reshape((height, width, 4))

        rgb = rgba[:, :, :3].astype(np.uint8)
        return rgb


    def compute_term_trunc_reward(self) -> None:
        """Compute reward with goal-seeking and obstacle avoidance"""
        super().compute_term_trunc_reward()

        if self.termination or self.truncation:
            return

        if len(self.waypoints.targets) == 0:
            return

        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = self.compute_attitude()
        goal_pos = self.waypoints.targets[0]

        horizontal_dist = np.linalg.norm(lin_pos[:2] - goal_pos[:2])
        vertical_dist = abs(lin_pos[2] - goal_pos[2])
        current_distance = np.linalg.norm(lin_pos - goal_pos)

        if not hasattr(self, 'previous_distance'):
            self.previous_distance = current_distance

        velocity = np.array(lin_vel).flatten()

        # reward progress toward goal
        progress = self.previous_distance - current_distance
        self.reward += 10.0 * np.clip(progress, -0.5, 0.5)
        self.previous_distance = current_distance

        # reward velocity toward goal
        goal_direction = goal_pos - lin_pos
        goal_direction_norm = goal_direction / (np.linalg.norm(goal_direction) + 1e-8)
        speed_toward_goal = np.dot(velocity, goal_direction_norm)
        self.reward += 2.0 * np.clip(speed_toward_goal, -2.0, 2.0)

        # reward proximity to goal
        proximity_reward = min(5.0 / (current_distance + 0.1), 12.0)
        self.reward += proximity_reward

        # penalty to prevent drone from flying straight into ground
        current_height = lin_pos[2]
        if current_height < 0.5:
            self.reward -= 10.0 * (0.5 - current_height)

        # terminate episode if drone hits the floor
        if current_height < 0.15:
            self.reward -= 50.0
            self.termination = True
            self.info["floor_crash"] = True
            return

        # height penalty if drone flies too high 
        goal_height = goal_pos[2]
        if current_height > goal_height + 4.0:
            self.reward -= 0.5 * (current_height - goal_height - 4.0)

        # time penalty
        self.reward -= self.time_step_penalty

        # terminate episode if drone crashes into tree
        if self._check_tree_collision():
            self.reward = -self.tree_collision_penalty
            self.termination = True
            self.info["tree_collision"] = True
            self.info["collision"] = True
            return

        # penalty if drone gets too close to obstacle
        obstacle_distances = self.state.get("obstacle_distances", None)
        if obstacle_distances is not None:
            min_distance = np.min(obstacle_distances)
            danger_radius = 2.0
            if min_distance < danger_radius:
                normalized = min_distance / danger_radius
                obstacle_penalty = self.tree_proximity_penalty_weight * (1.0 - normalized) ** 2
                self.reward -= obstacle_penalty

        # waypoint reached!
        if self.waypoints.target_reached:
            self.reward += 100.0
            self.waypoints.advance_targets()
            self.truncation |= self.waypoints.all_targets_reached