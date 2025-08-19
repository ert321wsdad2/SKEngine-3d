import os
import sys
import math
import argparse
from pathlib import Path

import numpy as np

from engine.camera import Camera
from engine.mesh import create_cube_mesh
from engine.math3d import transform_points


def clamp(value: float, min_value: float, max_value: float) -> float:
	return max(min_value, min(max_value, value))


def run_headless(width: int, height: int, frames: int) -> None:
	# Ensure headless video driver before importing pygame/renderer
	os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
	import pygame
	from engine.renderer import Renderer

	aspect_wh = width / float(height)
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

	print(f"Rendered {frames} frames to: {Path('output').resolve()}")


def run_interactive(width: int, height: int) -> None:
	# Use default SDL driver (windowed). Import pygame/renderer now.
	import pygame
	from engine.renderer import Renderer

	pygame.init()
	window = pygame.display.set_mode((width, height))
	pygame.display.set_caption("3D Proto - WASD + Mouse, Space to jump, Esc to quit")
	clock = pygame.time.Clock()

	# Mouse look setup
	pygame.event.set_grab(True)
	pygame.mouse.set_visible(False)
	pygame.mouse.get_rel()  # reset relative motion

	camera = Camera(
		position=np.array([0.0, 1.0, 0.0], dtype=np.float32),
		rotation_radians=(0.0, 0.0, 0.0),
		fov_degrees=70.0,
		near=0.1,
		far=200.0,
		aspect_wh=width / float(height),
	)

	renderer = Renderer(width, height)
	cube_vertices, cube_edges = create_cube_mesh(1.5)

	# Camera control state
	yaw = 0.0
	pitch = 0.0
	mouse_sensitivity = 0.003
	move_speed = 3.0
	sprint_multiplier = 1.8

	# Simple jump/physics
	gravity = -18.0
	jump_velocity = 7.0
	vertical_velocity = 0.0
	eye_height = 1.0
	ground_y = eye_height

	running = True
	angle = 0.0
	while running:
		dt = max(1.0 / 240.0, clock.tick(120) / 1000.0)

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
				running = False
			elif event.type == pygame.VIDEORESIZE:
				width, height = event.w, event.h
				window = pygame.display.set_mode((width, height))
				renderer = Renderer(width, height)
				camera.aspect_wh = width / float(height)

		# Mouse look (relative)
		dx, dy = pygame.mouse.get_rel()
		yaw += dx * mouse_sensitivity
		pitch += -dy * mouse_sensitivity
		pitch = clamp(pitch, -math.pi / 2.0 + 0.01, math.pi / 2.0 - 0.01)
		camera.rotation_radians = (pitch, yaw, 0.0)

		# Movement input
		keys = pygame.key.get_pressed()
		speed = move_speed * (sprint_multiplier if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 1.0)
		forward = np.array([math.sin(yaw), 0.0, math.cos(yaw)], dtype=np.float32)
		right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)], dtype=np.float32)

		move_dir = np.zeros(3, dtype=np.float32)
		if keys[pygame.K_w]:
			move_dir += forward
		if keys[pygame.K_s]:
			move_dir -= forward
		if keys[pygame.K_a]:
			move_dir -= right
		if keys[pygame.K_d]:
			move_dir += right

		# Normalize horizontal movement to prevent faster diagonal motion
		length = float(np.linalg.norm(move_dir))
		if length > 1e-5:
			move_dir /= length
		camera.position += move_dir * speed * dt

		# Jump and gravity
		grounded = camera.position[1] <= ground_y + 1e-4
		if grounded:
			camera.position[1] = ground_y
			vertical_velocity = 0.0
			if keys[pygame.K_SPACE]:
				vertical_velocity = jump_velocity
		else:
			vertical_velocity += gravity * dt
		camera.position[1] += vertical_velocity * dt

		# Animate cube in front of camera
		angle += dt
		world_vertices = transform_points(
			cube_vertices,
			(rotation := (angle * 0.6, angle * 0.9, angle * 0.3)),
			translation=np.array([0.0, 1.0, 4.0], dtype=np.float32),
		)

		# Render
		screen = pygame.display.get_surface()
		renderer.clear(screen, (12, 12, 16))
		renderer.draw_wireframe(screen, camera, world_vertices, cube_edges, (200, 240, 255))

		pygame.display.flip()
		pygame.display.set_caption(f"3D Proto - FPS: {clock.get_fps():.1f} | WASD + Mouse, Space")

	pygame.mouse.set_visible(True)
	pygame.event.set_grab(False)
	pygame.quit()


def main() -> None:
	parser = argparse.ArgumentParser(description="3D prototype: interactive or headless render")
	parser.add_argument("--headless", action="store_true", help="render frames to output/ without opening a window")
	parser.add_argument("--width", type=int, default=800)
	parser.add_argument("--height", type=int, default=600)
	parser.add_argument("--frames", type=int, default=120, help="frames to render in headless mode")
	args = parser.parse_args()

	if args.headless:
		run_headless(args.width, args.height, args.frames)
	else:
		run_interactive(args.width, args.height)


if __name__ == "__main__":
	main()