from typing import List, Tuple

import numpy as np
import pygame

from .math3d import ndc_to_screen
from .camera import Camera


Color = Tuple[int, int, int]


class Renderer:
	def __init__(self, width: int, height: int):
		self.width = width
		self.height = height

	def clear(self, surface: pygame.Surface, color: Color = (12, 12, 16)) -> None:
		surface.fill(color)

	def draw_wireframe(self, surface: pygame.Surface, camera: Camera, vertices_world: np.ndarray, edges: List[Tuple[int, int]], color: Color = (200, 220, 255)) -> None:
		camera_space = camera.world_to_camera(vertices_world)
		ndc, valid_mask = camera.project(camera_space)
		screen_xy = ndc_to_screen(ndc, self.width, self.height)

		for i0, i1 in edges:
			if not (valid_mask[i0] and valid_mask[i1]):
				continue
			p0 = screen_xy[i0]
			p1 = screen_xy[i1]
			pygame.draw.line(surface, color, p0, p1, 2)