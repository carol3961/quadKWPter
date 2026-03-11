import numpy as np
from typing import Any, Literal
from gymnasium import spaces
from PyFlyt.gym_envs.quadx_envs.quadx_base_env import QuadXBaseEnv
from PyFlyt.gym_envs.quadx_envs.quadx_waypoints_env import QuadXWaypointsEnv
import os
import pybullet as p


DISTANCE_PROGRESS_MAX_REWARD_PER_STEP = 10.0
VELOCITY_TOWARD_GOAL_MAX_REWARD_PER_STEP = 2.0

REWARD_PROXIMITY_MAX = 12.0
REWARD_PROXIMITY_BASE = 5.0

GROUND_AVOIDANCE_SCALE = 10.0
FLOOR_CRASH_PENALTY = 50.0

HEIGHT_PENALTY_SCALE = 0.5

WAYPOINT_REWARD_BONUS = 100.0
TREE_PROX_PENALTY_WEIGHT = 3.0
TIME_STEP_PENALTY = 0.05
TREE_COLLISION_PENALTY = 150.0

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
        max_duration_seconds: float = 30.0,
        angle_representation: Literal["euler", "quaternion"] = "quaternion",
        agent_hz: int = 30,
        render_mode: None | Literal["human", "rgb_array"] = None,
        render_resolution: tuple[int, int] = (480, 480),
        # Tree parameters
        num_trees: int = 20,
        tree_radius_range: tuple[float, float] = (0.2, 0.3),
        tree_height_range: tuple[float, float] = (1.0, 2.0),
        tree_mesh_dir_path: str = os.path.join(os.getcwd(), "gazebo_pine_tree_model", "meshes"),
        tree_collision_penalty: float = TREE_COLLISION_PENALTY,
        tree_proximity_penalty_weight: float = TREE_PROX_PENALTY_WEIGHT,
        goal_area: dict = None,
        # Sensor parameters
        num_sensors: int = 8,
        sensor_range: float = 5.0,
        # Time penalty
        time_step_penalty: float = TIME_STEP_PENALTY,
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
            far = 0.65 * flight_dome_size
            near = 0.50 * flight_dome_size
            self.goal_area = {
                'x_min': near, 'x_max': far,
                'y_min': near, 'y_max': far,
                'z_min': 1.5,  'z_max': 2.5,
            }
            # self.goal_area = {
            #     'x_min': 12.0, 'x_max': 16.0,
            #     'y_min': 12.0, 'y_max': 16.0,
            #     'z_min': 1.5,  'z_max': 2.5,
            # }
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
        self, *, seed: None | int = None, options: None | dict[str, Any] = None
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
        self.info["tree_collision"] = False
        self.info["episode_timeout"] = False
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

    def _check_tree_collision(self) -> bool:
        """Check if drone has collided with any tree"""

        # Get drone's body ID from the aviary
        try:
            drone_id = self.env.drones[0].Id
        except:
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
        """Generates trees in a corridor between drone start and waypoint(s).

        Always places NUM_BLOCKER_TREES directly on the straight-line path first,
        so the agent can never win by flying straight every episode.
        Remaining trees are placed randomly within the corridor as before.
        """

        self.tree_positions = []

        start = np.array(self.start_pos[0][:2], dtype=float)
        goal = np.array(self.waypoints.targets[0][:2], dtype=float)

        corridor_width = 5
        max_attempts = 100
        min_start_clearance = 2.0
        min_waypoint_clearance = 3.0

        direction = goal - start
        norm = np.linalg.norm(direction) + 1e-8
        direction_norm = direction / norm
        perpendicular = np.array([-direction_norm[1], direction_norm[0]])

        # Trees span full dome height so drone must always go around, never over
        height = float(self.flight_dome_size)

        def _spawn_tree(x: float, y: float, radius: float) -> None:
            """Creates physics + visual body and appends to self.tree_positions."""
            base_position = [x, y, height / 2]
            collision_shape = self.env.createCollisionShape(
                shapeType=self.env.GEOM_CYLINDER,
                radius=radius,
                height=height,
            )
            visual_shape = self.env.createVisualShape(
                shapeType=self.env.GEOM_CYLINDER,
                radius=radius,
                length=height,
                rgbaColor=[34/255, 100/255, 34/255, 1],
            )
            tree_id = self.env.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=base_position,
                baseOrientation=[0, 0, 0, 1],
            )
            self.tree_positions.append({
                "id": tree_id,
                "position": np.array([x, y, 0.0], dtype=float),
                "radius": radius,
                "height": height,
            })

        def _is_valid_position(pos: np.ndarray, radius: float) -> bool:
            """Shared validity checks for any candidate tree position."""
            x, y = float(pos[0]), float(pos[1])

            if abs(x) > self.flight_dome_size or abs(y) > self.flight_dome_size:
                return False

            if np.linalg.norm(pos[:2] - start) < min_start_clearance:
                return False

            if hasattr(self.waypoints, "targets"):
                for wp in self.waypoints.targets:
                    if np.linalg.norm(pos[:2] - np.array(wp[:2], dtype=float)) < min_waypoint_clearance:
                        return False

            for placed_tree in self.tree_positions:
                min_sep = radius + placed_tree["radius"] + 0.2
                if np.linalg.norm(pos[:2] - placed_tree["position"][:2]) < min_sep:
                    return False

            return True

        # ------------------------------------------------------------------
        # Phase 1 — Blocker trees: forced onto the straight-line path.
        #
        # We sample t in (0.25, 0.75) so blockers land in the middle stretch
        # (not right next to start or goal), and restrict the perpendicular
        # offset to ±0.3 m so the tree genuinely straddles the direct route.
        # We attempt up to max_attempts times per blocker; if placement fails
        # (very tight geometry) we skip rather than hang forever.
        # ------------------------------------------------------------------
        NUM_BLOCKERS = 2
        BLOCKER_PERPENDICULAR_BAND = 0.3   # metres either side of centre line
        # Space blockers evenly along the path so they don't cluster together
        blocker_t_slots = [(i + 1) / (NUM_BLOCKERS + 1) for i in range(NUM_BLOCKERS)]
        # Add a small random jitter so episodes aren't identical
        BLOCKER_T_JITTER = 0.08

        for t_center in blocker_t_slots:
            radius = float(self.np_random.uniform(*self.tree_radius_range))
            placed = False

            for _ in range(max_attempts):
                t = float(np.clip(
                    t_center + self.np_random.uniform(-BLOCKER_T_JITTER, BLOCKER_T_JITTER),
                    0.15, 0.85,
                ))
                center_point = start + t * (goal - start)
                offset = self.np_random.uniform(-BLOCKER_PERPENDICULAR_BAND, BLOCKER_PERPENDICULAR_BAND)
                xy_pos = center_point + offset * perpendicular
                pos = np.array([float(xy_pos[0]), float(xy_pos[1]), 0.0], dtype=float)

                if not _is_valid_position(pos, radius):
                    continue

                _spawn_tree(float(pos[0]), float(pos[1]), radius)
                placed = True
                break

            if not placed:
                print(f"[_generate_trees] Warning: could not place blocker tree (t≈{t_center:.2f}) after {max_attempts} attempts.")

        # ------------------------------------------------------------------
        # Phase 2 — Random corridor trees (same logic as before).
        # We subtract the blockers already placed so total tree count stays
        # equal to self.num_trees.
        # ------------------------------------------------------------------
        remaining = self.num_trees - len(self.tree_positions)

        for _ in range(remaining):
            radius = float(self.np_random.uniform(*self.tree_radius_range))
            placed = False

            for _ in range(max_attempts):
                t = self.np_random.uniform(0.0, 1.0)
                center_point = start + t * (goal - start)

                offset = self.np_random.uniform(-corridor_width, corridor_width)
                xy_pos = center_point + offset * perpendicular
                pos = np.array([float(xy_pos[0]), float(xy_pos[1]), 0.0], dtype=float)

                if not _is_valid_position(pos, radius):
                    continue

                _spawn_tree(float(pos[0]), float(pos[1]), radius)
                placed = True
                break

            if not placed:
                continue


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
        """Compute reward with goal-seeking and obstacle avoidance, with reward-component logging."""
        super().compute_term_trunc_reward()

        # Base reward from parent env
        base_reward = float(getattr(self, "reward", 0.0))

        # Initialize reward breakdown
        distance_progress_reward = 0.0
        velocity_toward_goal_reward = 0.0
        proximity_reward = 0.0
        ground_avoidance_penalty = 0.0
        height_penalty = 0.0
        time_penalty = 0.0
        tree_collision_penalty = 0.0
        obstacle_proximity_penalty = 0.0
        floor_collision_penalty = 0.0

        if self.truncation and not self.termination and not self.info.get("env_complete", False):
            self.info["episode_timeout"] = True

        if self.termination or self.truncation:
            self.info["reward_total"] = float(self.reward)
            self.info["reward_base"] = base_reward
            self.info["reward_distance_progress"] = distance_progress_reward
            self.info["reward_velocity_toward_goal"] = velocity_toward_goal_reward
            self.info["reward_goal_proximity"] = proximity_reward
            self.info["reward_ground_avoidance_penalty"] = ground_avoidance_penalty
            self.info["reward_height_penalty"] = height_penalty
            self.info["reward_time_penalty"] = time_penalty
            self.info["reward_tree_collision_penalty"] = tree_collision_penalty
            self.info["reward_floor_collision_penalty"] = floor_collision_penalty
            self.info["reward_obstacle_proximity_penalty"] = obstacle_proximity_penalty
            return

        if len(self.waypoints.targets) == 0:
            self.info["reward_total"] = float(self.reward)
            self.info["reward_base"] = base_reward
            return

        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = self.compute_attitude()
        goal_pos = self.waypoints.targets[0]

        current_distance = np.linalg.norm(lin_pos - goal_pos)

        if not hasattr(self, "previous_distance"):
            self.previous_distance = current_distance

        velocity = np.array(lin_vel).flatten()

        # 1. reward progress toward goal
        progress = self.previous_distance - current_distance
        distance_progress_reward = 7.0 * np.clip(progress, -1.0, 0.5)
        self.reward += distance_progress_reward
        self.previous_distance = current_distance
        # 2. reward velocity toward goal
        # goal_direction = goal_pos - lin_pos
        # goal_direction_norm = goal_direction / (np.linalg.norm(goal_direction) + 1e-8)
        # speed_toward_goal = np.dot(velocity, goal_direction_norm)
        # velocity_toward_goal_reward = 1.0 * np.clip(speed_toward_goal, -2.0, 2.0)
        # self.reward += velocity_toward_goal_reward

        # 3. reward proximity to goal
        proximity_reward = min(3.0 / (current_distance + 0.1), 5.0)
        self.reward += proximity_reward

        # 4. penalty to prevent drone from flying straight into ground
        current_height = lin_pos[2]
        if current_height < 0.5:
            ground_avoidance_penalty = 5.0 * (0.5 - current_height)
            self.reward -= ground_avoidance_penalty

        # terminate episode if drone hits the floor
        # if current_height < 0.15:
        #     floor_collision_penalty = 50.0
        #     self.reward -= floor_collision_penalty
        #     self.termination = True
            # self.info["floor_crash"] = True
            # self.info["collision"] = True

        #     self.info["reward_total"] = float(self.reward)
        #     self.info["reward_base"] = base_reward
        #     self.info["reward_distance_progress"] = distance_progress_reward
        #     self.info["reward_velocity_toward_goal"] = velocity_toward_goal_reward
        #     self.info["reward_goal_proximity"] = proximity_reward
        #     self.info["reward_ground_avoidance_penalty"] = ground_avoidance_penalty
        #     self.info["reward_height_penalty"] = height_penalty
        #     self.info["reward_time_penalty"] = time_penalty
        #     self.info["reward_tree_collision_penalty"] = tree_collision_penalty
        #     self.info["reward_floor_collision_penalty"] = floor_collision_penalty
        #     self.info["reward_obstacle_proximity_penalty"] = obstacle_proximity_penalty
        #     return

        # 5. height penalty if drone flies too high
        # goal_height = goal_pos[2]
        # if current_height > goal_height + 4.0:
        #     height_penalty = 0.5 * (current_height - goal_height - 4.0)
        #     self.reward -= height_penalty

        # 6. time penalty
        time_penalty = self.time_step_penalty
        self.reward -= time_penalty

        # terminate episode if drone crashes into tree
        if self._check_tree_collision():
            tree_collision_penalty = float(self.tree_collision_penalty)
            self.reward = -tree_collision_penalty
            self.termination = True
            self.info["tree_collision"] = True

            self.info["reward_total"] = float(self.reward)
            self.info["reward_base"] = base_reward
            self.info["reward_distance_progress"] = distance_progress_reward
            self.info["reward_velocity_toward_goal"] = velocity_toward_goal_reward
            self.info["reward_goal_proximity"] = proximity_reward
            self.info["reward_ground_avoidance_penalty"] = ground_avoidance_penalty
            self.info["reward_height_penalty"] = height_penalty
            self.info["reward_time_penalty"] = time_penalty
            self.info["reward_tree_collision_penalty"] = tree_collision_penalty
            self.info["reward_floor_collision_penalty"] = floor_collision_penalty
            self.info["reward_obstacle_proximity_penalty"] = obstacle_proximity_penalty
            return

        # 8. penalty if drone gets too close to obstacle
        # obstacle_distances = self.state.get("obstacle_distances", None)
        # if obstacle_distances is not None:
        #     min_distance = np.min(obstacle_distances)
        #     danger_radius = 2.0
        #     if min_distance < danger_radius:
        #         normalized = min_distance / danger_radius
        #         obstacle_proximity_penalty = self.tree_proximity_penalty_weight * (1.0 - normalized) ** 2
        #         self.reward -= obstacle_proximity_penalty



        # 9. waypoint reached
        # if self.waypoints.target_reached:
        #     self.reward += 100.0
        #     self.waypoints.advance_targets()
        #     if self.waypoints.all_targets_reached:
        #         self.truncation = True
        #         self.info["env_complete"] = True

        # 10 boundary warning penalty
        # dist_from_origin = np.linalg.norm(lin_pos)
        # boundary_margin = 0.85 * self.flight_dome_size  # warn at 85% of dome radius
        # if dist_from_origin > boundary_margin:
        #     boundary_penalty = 5.0 * (dist_from_origin - boundary_margin)
        #     self.reward -= boundary_penalty

        # Final reward logging
        self.info["reward_total"] = float(self.reward)
        self.info["reward_base"] = base_reward
        self.info["reward_distance_progress"] = distance_progress_reward
        self.info["reward_velocity_toward_goal"] = velocity_toward_goal_reward
        self.info["reward_goal_proximity"] = proximity_reward
        self.info["reward_ground_avoidance_penalty"] = ground_avoidance_penalty
        self.info["reward_height_penalty"] = height_penalty
        self.info["reward_time_penalty"] = time_penalty
        self.info["reward_tree_collision_penalty"] = tree_collision_penalty
        self.info["reward_floor_collision_penalty"] = floor_collision_penalty
        self.info["reward_obstacle_proximity_penalty"] = obstacle_proximity_penalty
