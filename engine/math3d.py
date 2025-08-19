import math
from typing import Tuple

import numpy as np


def build_rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
	"""Return 3x3 rotation matrix from Euler angles (radians). Order Z * Y * X."""
	cz, sz = math.cos(rz), math.sin(rz)
	cy, sy = math.cos(ry), math.sin(ry)
	cx, sx = math.cos(rx), math.sin(rx)

	Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
	Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
	Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
	return (Rz @ Ry @ Rx).astype(np.float32)


def transform_points(points: np.ndarray, rotation_radians: Tuple[float, float, float], translation: np.ndarray) -> np.ndarray:
	"""Apply rotation and translation to Nx3 points."""
	R = build_rotation_matrix(rotation_radians[0], rotation_radians[1], rotation_radians[2])
	rotated = (R @ points.T).T
	return rotated + translation.reshape(1, 3)


def inverse_transform_points(points: np.ndarray, rotation_radians: Tuple[float, float, float], translation: np.ndarray) -> np.ndarray:
	"""Apply inverse of rotation/translation to convert world points into camera space."""
	R = build_rotation_matrix(rotation_radians[0], rotation_radians[1], rotation_radians[2])
	R_inv = R.T
	shifted = points - translation.reshape(1, 3)
	return (R_inv @ shifted.T).T


def project_points_perspective(points_camera: np.ndarray, fov_degrees: float, aspect_wh: float, z_near: float):
	"""
	Project camera-space Nx3 points to normalized device coords and return mask of valid points (z > z_near).
	NDC range is approximately [-1, 1] for x and y when within view.
	"""
	f = 1.0 / math.tan(math.radians(fov_degrees) * 0.5)
	z = points_camera[:, 2]
	valid = z > z_near
	x_ndc = (points_camera[:, 0] * f / aspect_wh) / z
	y_ndc = (points_camera[:, 1] * f) / z
	ndc = np.stack([x_ndc, y_ndc], axis=1)
	return ndc, valid


def ndc_to_screen(ndc_xy: np.ndarray, width: int, height: int) -> np.ndarray:
	"""Map NDC [-1,1] to pixel coordinates."""
	half_w = width * 0.5
	half_h = height * 0.5
	x = (ndc_xy[:, 0] + 1.0) * half_w
	y = (1.0 - ndc_xy[:, 1]) * half_h
	return np.stack([x, y], axis=1).astype(np.int32)