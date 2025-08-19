import os
import math
from pathlib import Path

# Headless rendering for environments without a display
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame

from engine.camera import Camera
from engine.mesh import create_cube_mesh
from engine.renderer import Renderer
from engine.math3d import transform_points


def main() -> None:
	width, height = 800, 600
	aspect_wh = width / float(height)
	frames = 120
	output_dir = Path("output")
	output_dir.mkdir(parents=True, exist_ok=True)

	pygame.init()
	surface = pygame.Surface((width, height))

	camera = Camera(
		position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
		rotation_radians=(0.0, 0.0, 0.0),
		fov_degrees=70.0,
		near=0.1,
		far=100.0,
		aspect_wh=aspect_wh,
	)

	renderer = Renderer(width, height)
	cube_vertices, cube_edges = create_cube_mesh(1.5)

	for frame_idx in range(frames):
		angle = (frame_idx / frames) * 2.0 * math.pi
		world_vertices = transform_points(
			cube_vertices,
			(rotation := (angle * 0.6, angle * 0.9, angle * 0.3)),
			translation=np.array([math.sin(angle) * 0.5, math.sin(angle * 0.7) * 0.2, 3.0], dtype=np.float32),
		)

		renderer.clear(surface, (12, 12, 16))
		renderer.draw_wireframe(surface, camera, world_vertices, cube_edges, (200, 240, 255))

		out_path = output_dir / f"frame_{frame_idx:03d}.png"
		pygame.image.save(surface, out_path.as_posix())

	print(f"Rendered {frames} frames to: {output_dir.resolve()}")


if __name__ == "__main__":
	main()