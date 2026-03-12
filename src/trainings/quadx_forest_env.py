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
        # Blocking trees — placed directly on the start→goal line
        num_blocking_trees: int = 1,
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
        self.num_blocking_trees = num_blocking_trees

        # Sensor configuration
        self.num_sensors = num_sensors
        self.sensor_range = sensor_range

        # Default goal area
        if goal_area is None:
            self.goal_area = {
                "x_min": 5.0, "x_max": 8.0,
                "y_min": 5.0, "y_max": 8.0,
                "z_min": 1.5, "z_max": 2.5,
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
        self,
        *,
        seed: None | int = None,
        options: None | dict[str, Any] = dict(),
    ) -> tuple[dict[Literal["attitude", "target_deltas", "obstacle_distances"], np.ndarray], dict]:
        """Resets the environment with trees and custom spawn positions."""

        QuadXBaseEnv.begin_reset(self, seed, options)

        # Re-initialise waypoints
        self.waypoints.reset(self.env, self.np_random)

        # Place goal in constrained area
        goal_x = self.np_random.uniform(self.goal_area["x_min"], self.goal_area["x_max"])
        goal_y = self.np_random.uniform(self.goal_area["y_min"], self.goal_area["y_max"])
        goal_z = self.np_random.uniform(self.goal_area["z_min"], self.goal_area["z_max"])
        desired_goal = np.array([goal_x, goal_y, goal_z])

        self.waypoints.targets[0] = desired_goal

        if self.waypoints.enable_render and len(self.waypoints.target_visual) > 0:
            self.env.resetBasePositionAndOrientation(
                self.waypoints.target_visual[0],
                desired_goal.tolist(),
                [0, 0, 0, 1],
            )

        # Generate trees (blocking + random)
        self._generate_trees()

        # Initialise distance tracking
        self.previous_distance = float(np.linalg.norm(self.start_pos[0] - desired_goal))

        self.info["num_targets_reached"] = 0

        QuadXBaseEnv.end_reset(self)

        return self.state, self.info

    def compute_state(self) -> None:
        """Computes the state including obstacle sensor readings."""
        super().compute_state()
        self.state["obstacle_distances"] = self._get_obstacle_distances()

    def _get_obstacle_distances(self) -> np.ndarray:
        """Cast rays around the drone to detect obstacles."""
        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = self.compute_attitude()

        rotation_matrix = np.array(
            self.env.getMatrixFromQuaternion(list(quaternion))
        ).reshape(3, 3)

        drone_pos = np.array(lin_pos).flatten()
        distances = np.full(self.num_sensors, self.sensor_range, dtype=np.float64)

        for i in range(self.num_sensors):
            angle     = 2 * np.pi * i / self.num_sensors
            local_dir = np.array([np.cos(angle), np.sin(angle), 0.0])
            world_dir = rotation_matrix @ local_dir

            ray_from = drone_pos
            ray_to   = ray_from + world_dir * self.sensor_range

            ray_result = self.env.rayTest(ray_from.tolist(), ray_to.tolist())

            if len(ray_result) > 0 and ray_result[0][0] >= 0:
                distances[i] = ray_result[0][2] * self.sensor_range

        return distances

    # ──────────────────────────────────────────────────────────────────────────
    # Collision helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _check_tree_collision(self) -> bool:
        """Check if the drone has collided with any tree."""
        try:
            drone_id = self.env.drones[0].Id
        except Exception:
            return False

        contact_points = self.env.getContactPoints(bodyA=drone_id)
        tree_ids = {tree["id"] for tree in self.tree_positions}

        for contact in contact_points:
            if contact[2] in tree_ids:
                return True

        return False

    def _get_tree_mesh_path(self):
        """Finds the mesh file for the pine tree model."""
        if os.path.exists(self.tree_mesh_dir_path):
            mesh_files = [
                f for f in os.listdir(self.tree_mesh_dir_path)
                if f.endswith((".dae", ".obj", ".stl"))
            ]
            if mesh_files:
                return os.path.join(self.tree_mesh_dir_path, mesh_files[0])
        raise FileNotFoundError(f"No mesh files found in {self.tree_mesh_dir_path}")

    def _spawn_tree(self, x: float, y: float) -> None:
        """Creates a single tree at (x, y) and appends it to self.tree_positions."""
        height = float(self.np_random.uniform(*self.tree_height_range))
        radius = float(self.np_random.uniform(*self.tree_radius_range))
        orient = p.getQuaternionFromEuler([0, 0, float(self.np_random.uniform(0, 2 * np.pi))])

        visual_shape = self.env.createVisualShape(
            shapeType=self.env.GEOM_MESH,
            fileName=self.tree_mesh_path,
            meshScale=[height, height, height],
            rgbaColor=[178 / 255, 172 / 255, 136 / 255, 1],
        )
        collision_shape = self.env.createCollisionShape(
            shapeType=self.env.GEOM_CYLINDER,
            radius=radius,
            height=height*10,
        )
        tree_id = self.env.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[x, y, 0.0],
            baseOrientation=orient,
        )
        self.tree_positions.append({
            "id":       tree_id,
            "position": np.array([x, y, 0.0]),
            "radius":   radius,
            "height":   height,
        })

    def _generate_trees(self):
        """
        Generates trees in two passes:

        Pass 1 — Blocking trees
            Evenly spaced along the straight line from the drone spawn point to
            the goal. These are guaranteed to sit directly in the drone's path,
            forcing it to navigate around them.

            Example with num_blocking_trees=3:
                t = 0.25, 0.50, 0.75  →  quarter-way, mid-way, three-quarter-way

        Pass 2 — Random background trees
            Placed anywhere in the dome, respecting minimum clearance from the
            spawn point and goal.
        """
        self.tree_positions = []

        start_2d = np.array(self.start_pos[0][:2], dtype=float)
        goal_2d  = np.array(self.waypoints.targets[0][:2], dtype=float)

        # ── Pass 1: blocking trees evenly spaced on the start → goal line ─────
        n = self.num_blocking_trees
        for i in range(1, n + 1):
            t   = i / (n + 1)                          # e.g. 0.25, 0.50, 0.75
            mid = start_2d + t * (goal_2d - start_2d)
            self._spawn_tree(float(mid[0]), float(mid[1]))

        # ── Pass 2: random background trees ───────────────────────────────────
        min_start_clearance    = 2.0
        min_waypoint_clearance = 2.0

        for _ in range(self.num_trees):
            attempts          = 0
            is_valid_position = False

            while attempts < 50:
                x   = float(self.np_random.uniform(-self.flight_dome_size, self.flight_dome_size))
                y   = float(self.np_random.uniform(-self.flight_dome_size, self.flight_dome_size))
                pos = np.array([x, y])

                # Clear the spawn zone
                if np.linalg.norm(pos - start_2d) < min_start_clearance:
                    attempts += 1
                    continue

                # Clear all waypoints
                too_close = False
                if hasattr(self.waypoints, "targets"):
                    for wp in self.waypoints.targets:
                        if np.linalg.norm(pos - np.array(wp[:2], dtype=float)) < min_waypoint_clearance:
                            too_close = True
                            break
                if too_close:
                    attempts += 1
                    continue

                is_valid_position = True
                break

            if not is_valid_position:
                continue

            self._spawn_tree(x, y)


    def render(self):
        """Third-person chase camera render."""
        if getattr(self, "render_mode", None) != "rgb_array":
            try:
                return super().render()
            except Exception:
                return None

        width, height = getattr(self, "render_resolution", (480, 480))
        _, _, _, lin_pos, _ = self.compute_attitude()
        target = np.array(lin_pos).flatten().tolist()

        view_matrix = self.env.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=target,
            distance=6.0,
            yaw=45,
            pitch=-30,
            roll=0,
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

        return rgba[:, :, :3].astype(np.uint8)

    # ──────────────────────────────────────────────────────────────────────────
    # Reward
    # ──────────────────────────────────────────────────────────────────────────

    def compute_term_trunc_reward(self) -> None:
        """Compute reward with goal-seeking and obstacle avoidance."""
        super().compute_term_trunc_reward()

        if self.termination or self.truncation:
            return

        if len(self.waypoints.targets) == 0:
            return

        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = self.compute_attitude()
        goal_pos         = self.waypoints.targets[0]
        current_distance = float(np.linalg.norm(lin_pos - goal_pos))
        current_height   = float(lin_pos[2])

        if not hasattr(self, "previous_distance"):
            self.previous_distance = current_distance

        velocity = np.array(lin_vel).flatten()

        # ── 1. Progress toward goal ───────────────────────────────────────────
        progress = self.previous_distance - current_distance
        self.reward += 10.0 * np.clip(progress, -0.5, 0.5)
        self.previous_distance = current_distance

        # ── 2. Velocity toward goal ───────────────────────────────────────────
        goal_direction      = goal_pos - lin_pos
        goal_direction_norm = goal_direction / (np.linalg.norm(goal_direction) + 1e-8)
        speed_toward_goal   = np.dot(velocity, goal_direction_norm)
        self.reward += 2.0 * np.clip(speed_toward_goal, -2.0, 2.0)

        # ── 3. Proximity to goal ──────────────────────────────────────────────
        proximity_reward = min(5.0 / (current_distance + 0.1), 12.0)
        self.reward += proximity_reward

        # ── 4. Ground avoidance (soft) ────────────────────────────────────────
        if current_height < 0.5:
            self.reward -= 10.0 * (0.5 - current_height)

        # ── 5. Floor crash ────────────────────────────────────────────────────
        if current_height < 0.15:
            self.reward -= 50.0
            self.termination = True
            self.info["floor_crash"] = True
            return

        # ── 6. Height penalty if flying too high ──────────────────────────────
        goal_height = float(goal_pos[2])
        if current_height > goal_height + 4.0:
            self.reward -= 0.5 * (current_height - goal_height - 4.0)

        # ── 7. Time penalty ───────────────────────────────────────────────────
        self.reward -= self.time_step_penalty

        # ── 8. Tree collision ─────────────────────────────────────────────────
        if self._check_tree_collision():
            self.reward      = -self.tree_collision_penalty
            self.termination = True
            self.info["tree_collision"] = True
            self.info["collision"]      = True
            return

        # ── 9. Obstacle proximity penalty ─────────────────────────────────────
        obstacle_distances = self.state.get("obstacle_distances", None)
        if obstacle_distances is not None:
            min_distance  = float(np.min(obstacle_distances))
            danger_radius = 2.0
            if min_distance < danger_radius:
                normalized       = min_distance / danger_radius
                obstacle_penalty = self.tree_proximity_penalty_weight * (1.0 - normalized) ** 2
                self.reward     -= obstacle_penalty

        # ── 10. Waypoint reached ──────────────────────────────────────────────
        if self.waypoints.target_reached:
            self.reward += 100.0
            self.waypoints.advance_targets()
            self.truncation |= self.waypoints.all_targets_reached