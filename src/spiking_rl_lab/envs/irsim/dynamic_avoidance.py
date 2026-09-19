"""Dynamic obstacle avoidance task backed by IR-SIM."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from spiking_rl_lab.envs.curriculum import Curriculum

if TYPE_CHECKING:
    from irsim.env import EnvBase

    from spiking_rl_lab.trainers.validator import ValidationResult


class IRSimDynamicAvoidance(gym.Env[np.ndarray, np.ndarray], Curriculum):
    """Drive a differential robot to its goal through moving obstacles."""

    GOAL_DISTANCE_LIMIT = 10.0
    PROGRESS_REWARD_SCALE = 2.0
    STEP_PENALTY = 0.0125
    CLEARANCE_THRESHOLD = 0.6
    CLEARANCE_PENALTY_SCALE = 0.04
    ANGULAR_VELOCITY_PENALTY_SCALE = 0.005
    ANGULAR_VELOCITY_CHANGE_PENALTY_SCALE = 0.004
    SUCCESS_REWARD = 20.0
    COLLISION_PENALTY = 20.0
    SUCCESS_THRESHOLD = 0.8
    DIFFICULTY_STEP = 0.2

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["human"], "render_fps": 20}

    def __init__(
        self,
        *,
        render_mode: str | None = None,
        world_path: str | Path | None = None,
    ) -> None:
        """Create the task environment."""
        if render_mode not in {None, "human"}:
            msg = f"Unsupported render mode: {render_mode}"
            raise ValueError(msg)

        self.render_mode = render_mode
        self._world_path = (
            Path(world_path)
            if world_path is not None
            else Path(__file__).with_name("worlds") / "dynamic_avoidance.yaml"
        )
        self._sim: EnvBase | None = None
        self._previous_distance = 0.0
        self._previous_angular_velocity = 0.0
        self._difficulty = 1.0
        self._create_simulator(seed=None)

    def reset_curriculum(self) -> None:
        """Start training without random obstacles."""
        self._difficulty = 0.0

    def update_curriculum(self, result: ValidationResult) -> bool:
        """Advance difficulty when validation success reaches the required rate."""
        if self._difficulty >= 1.0:
            return False

        successes = result.metrics.get("success")
        if not successes:
            msg = "Validation result has no curriculum metric 'success'"
            raise RuntimeError(msg)
        success_rate = sum(float(success) for success in successes) / len(successes)
        if success_rate < self.SUCCESS_THRESHOLD:
            return False

        self._difficulty = min(1.0, self._difficulty + self.DIFFICULTY_STEP)
        return True

    def maximize_curriculum(self) -> None:
        """Use the complete obstacle scene for evaluation and demonstration."""
        self._difficulty = 1.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the scenario and return its initial observation."""
        super().reset(seed=seed, options=options)
        if self._sim is None or seed is not None:
            self._create_simulator(seed)
        else:
            self._sim.reset(random=True)

        self._apply_difficulty(self._difficulty)
        self._previous_distance = self._distance_to_goal()
        self._previous_angular_velocity = 0.0
        return self._observation(), self._info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply a differential-drive command for one simulator step."""
        if self._sim is None:
            msg = "reset() must be called before step()"
            raise RuntimeError(msg)

        command = np.clip(
            np.asarray(action, dtype=np.float32),
            self.action_space.low,
            self.action_space.high,
        )
        self._sim.step(command)

        robot = self._sim.robot
        distance = self._distance_to_goal()
        terminated = bool(robot.arrive or robot.collision)
        reward = self._reward(
            distance,
            command,
            arrived=robot.arrive,
            collided=robot.collision,
        )
        self._previous_distance = distance
        self._previous_angular_velocity = float(command[1])

        if self.render_mode == "human":
            self.render()
        return self._observation(), reward, terminated, False, self._info()

    def render(self) -> None:
        """Render the current simulator state when human rendering is enabled."""
        if self.render_mode == "human" and self._sim is not None:
            self._sim.render()

    def close(self) -> None:
        """Close the underlying IR-SIM environment."""
        if self._sim is not None:
            self._sim.end(ending_time=0.001)
            self._sim = None

    def _create_simulator(self, seed: int | None) -> None:
        """Create a simulator instance for the packaged world."""
        import irsim

        if self._sim is not None:
            self.close()

        render = self.render_mode == "human"
        self._sim = irsim.make(
            str(self._world_path),
            headless=not render,
            display=render,
            log_level="ERROR",
            seed=seed,
        )

        robot = self._sim.robot
        velocity_min = np.asarray(robot.vel_min, dtype=np.float32).reshape(-1)
        velocity_max = np.asarray(robot.vel_max, dtype=np.float32).reshape(-1)
        self.action_space = spaces.Box(velocity_min, velocity_max, dtype=np.float32)

        scan = self._sim.get_lidar_scan()
        lidar_beams = len(scan["ranges"])
        self.observation_space = spaces.Box(
            low=np.concatenate(
                (
                    np.full(lidar_beams, scan["range_min"], dtype=np.float32),
                    np.array([0.0, -1.0, -1.0], dtype=np.float32),
                    velocity_min,
                )
            ),
            high=np.concatenate(
                (
                    np.full(lidar_beams, scan["range_max"], dtype=np.float32),
                    np.array([self.GOAL_DISTANCE_LIMIT, 1.0, 1.0], dtype=np.float32),
                    velocity_max,
                )
            ),
            dtype=np.float32,
        )

    def _apply_difficulty(self, difficulty: float) -> None:
        """Reduce the freshly reset full scene to the requested difficulty."""
        delete_ids: list[int] = []
        for group_name in ("dynamic_sfm", "dynamic_rvo", "static_random"):
            objects = self._sim.get_group_by_name(group_name)
            keep_count = int(len(objects) * difficulty)
            delete_ids.extend(obj.id for obj in objects[keep_count:])

        if delete_ids:
            self._sim.delete_objects(delete_ids)
        self._sim.refresh()

    def _observation(self) -> np.ndarray:
        """Build a robot-centric observation from LiDAR, goal, and velocity."""
        robot = self._sim.robot
        state = np.asarray(robot.state).reshape(-1)
        goal = np.asarray(robot.goal).reshape(-1)
        velocity = np.asarray(robot.velocity).reshape(-1)
        scan = self._sim.get_lidar_scan()
        lidar = np.asarray(scan["ranges"], dtype=np.float32)

        goal_delta = goal[:2] - state[:2]
        distance = float(np.linalg.norm(goal_delta))
        bearing = np.arctan2(goal_delta[1], goal_delta[0]) - state[2]
        navigation = np.array(
            [
                min(distance, self.GOAL_DISTANCE_LIMIT),
                np.sin(bearing),
                np.cos(bearing),
                velocity[0],
                velocity[1],
            ],
            dtype=np.float32,
        )
        return np.concatenate((lidar, navigation), dtype=np.float32)

    def _distance_to_goal(self) -> float:
        """Return planar distance from the robot to its current goal."""
        robot = self._sim.robot
        state = np.asarray(robot.state).reshape(-1)
        goal = np.asarray(robot.goal).reshape(-1)
        return float(np.linalg.norm(goal[:2] - state[:2]))

    def _reward(
        self,
        distance: float,
        action: np.ndarray,
        *,
        arrived: bool,
        collided: bool,
    ) -> float:
        """Reward progress while discouraging collision and unsafe clearance."""
        progress = self._previous_distance - distance
        min_range = float(np.min(self._sim.get_lidar_scan()["ranges"]))
        clearance_penalty = max(
            0.0,
            (self.CLEARANCE_THRESHOLD - min_range) / self.CLEARANCE_THRESHOLD,
        )
        reward = (
            self.PROGRESS_REWARD_SCALE * progress
            - self.STEP_PENALTY
            - self.CLEARANCE_PENALTY_SCALE * clearance_penalty
        )
        reward -= self.ANGULAR_VELOCITY_PENALTY_SCALE * abs(float(action[1]))
        reward -= self.ANGULAR_VELOCITY_CHANGE_PENALTY_SCALE * abs(
            float(action[1]) - self._previous_angular_velocity
        )
        if arrived:
            reward += self.SUCCESS_REWARD
        if collided:
            reward -= self.COLLISION_PENALTY
        return float(reward)

    def _info(self) -> dict[str, Any]:
        """Expose compact task diagnostics."""
        robot = self._sim.robot
        return {
            "distance_to_goal": self._distance_to_goal(),
            "success": bool(robot.arrive),
            "collision": bool(robot.collision),
            "difficulty": self._difficulty,
        }
