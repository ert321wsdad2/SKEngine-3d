from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .math3d import inverse_transform_points, project_points_perspective


@dataclass
class Camera:
	position: np.ndarray
	rotation_radians: Tuple[float, float, float]
	fov_degrees: float = 70.0
	near: float = 0.1
	far: float = 100.0
	aspect_wh: float = 1.0

	def world_to_camera(self, world_points: np.ndarray) -> np.ndarray:
		return inverse_transform_points(world_points, self.rotation_radians, self.position)

	def project(self, camera_points: np.ndarray):
		return project_points_perspective(camera_points, self.fov_degrees, self.aspect_wh, self.near)