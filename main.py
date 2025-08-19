import os
import sys
import math
import argparse
from pathlib import Path

import numpy as np

from engine.camera import Camera
from engine.mesh import create_cube_mesh, create_grid_mesh
from engine.math3d import transform_points


def clamp(value: float, min_value: float, max_value: float) -> float:
	return max(min_value, min(max_value, value))


def lerp(a: float, b: float, t: float) -> float:
	return a + (b - a) * t


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

	from engine.renderer import Renderer
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


def resolve_horizontal_collisions(prev_pos: np.ndarray, desired_pos: np.ndarray, aabbs: list, radius: float, camera_y: float) -> np.ndarray:
	"""
	Resolve collisions in XZ plane with expanded AABBs. Y is left unchanged.
	Only collides if camera_y is within the vertical span of the AABB (with small tolerance).
	"""
	new_pos = desired_pos.copy()
	for axis in (0, 2):  # 0: X, 2: Z
		for (mn, mx) in aabbs:
			# Expand AABB in XZ by radius
			mn_exp = mn.copy()
			mx_exp = mx.copy()
			mn_exp[0] -= radius
			mx_exp[0] += radius
			mn_exp[2] -= radius
			mx_exp[2] += radius

			# Check vertical overlap around camera y
			if not (camera_y >= mn[1] - 0.5 and camera_y <= mx[1] + 0.5):
				continue

			inside_x = (new_pos[0] >= mn_exp[0]) and (new_pos[0] <= mx_exp[0])
			inside_z = (new_pos[2] >= mn_exp[2]) and (new_pos[2] <= mx_exp[2])
			if inside_x and inside_z:
				if axis == 0:
					# Resolve along X toward nearest boundary based on prev_pos
					if prev_pos[0] <= mn_exp[0]:
						new_pos[0] = mn_exp[0]
					elif prev_pos[0] >= mx_exp[0]:
						new_pos[0] = mx_exp[0]
					else:
						new_pos[0] = mn_exp[0] if (new_pos[0] - mn_exp[0]) < (mx_exp[0] - new_pos[0]) else mx_exp[0]
				else:  # axis == 2
					if prev_pos[2] <= mn_exp[2]:
						new_pos[2] = mn_exp[2]
					elif prev_pos[2] >= mx_exp[2]:
						new_pos[2] = mx_exp[2]
					else:
						new_pos[2] = mn_exp[2] if (new_pos[2] - mn_exp[2]) < (mx_exp[2] - new_pos[2]) else mx_exp[2]
	return new_pos


def run_interactive(width: int, height: int, args) -> None:
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

	# Scene: ground grid and multiple static cubes (AABBs for collisions)
	grid_vertices, grid_edges = create_grid_mesh(width=80, depth=80, step=1.0, y=0.0)
	cube_verts_base, cube_edges = create_cube_mesh(1.5)
	cube_half = 0.75
	static_cubes = [
		(np.array([2.0, cube_half, 4.0], dtype=np.float32)),
		(np.array([-3.0, cube_half, 7.0], dtype=np.float32)),
		(np.array([0.0, cube_half, 10.0], dtype=np.float32)),
		(np.array([4.0, cube_half, -2.0], dtype=np.float32)),
	]
	# Precompute world vertices for static cubes
	static_cube_vertices_world = [cube_verts_base + pos.reshape(1, 3) for pos in static_cubes]
	# AABBs for collisions (min,max)
	obstacle_aabbs = []
	for pos in static_cubes:
		mn = np.array([pos[0] - cube_half, pos[1] - cube_half, pos[2] - cube_half], dtype=np.float32)
		mx = np.array([pos[0] + cube_half, pos[1] + cube_half, pos[2] + cube_half], dtype=np.float32)
		obstacle_aabbs.append((mn, mx))

	# Also add a bigger box as a wall
	wall_center = np.array([0.0, 1.0, -6.0], dtype=np.float32)
	wall_half = np.array([5.0, 1.0, 0.5], dtype=np.float32)
	wall_mn = wall_center - wall_half
	wall_max = wall_center + wall_half
	obstacle_aabbs.append((wall_mn, wall_max))

	# Camera control state
	target_yaw = 0.0
	target_pitch = 0.0
	yaw = 0.0
	pitch = 0.0
	mouse_sensitivity = float(args.mouse_sensitivity)
	look_smooth = clamp(float(args.look_smooth), 0.0, 0.999)
	move_speed = float(args.move_speed)
	sprint_multiplier = float(args.sprint_multiplier)
	camera_radius = float(args.camera_radius)

	# Simple jump/physics
	gravity = float(args.gravity)
	jump_velocity = float(args.jump_velocity)
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
		target_yaw += dx * mouse_sensitivity
		target_pitch += -dy * mouse_sensitivity
		target_pitch = clamp(target_pitch, -math.pi / 2.0 + 0.01, math.pi / 2.0 - 0.01)

		# Exponential smoothing toward target (0=no smoothing, 1=very smooth/slow)
		alpha = 1.0 - pow(1.0 - look_smooth, dt * 60.0)
		yaw = lerp(yaw, target_yaw, alpha)
		pitch = lerp(pitch, target_pitch, alpha)

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

		# Normalize horizontal movement
		length = float(np.linalg.norm(move_dir))
		if length > 1e-5:
			move_dir /= length

		# Integrate horizontal movement with collision resolution
		prev_pos = camera.position.copy()
		desired = camera.position + move_dir * speed * dt
		desired[1] = camera.position[1]
		camera.position = resolve_horizontal_collisions(prev_pos, desired, obstacle_aabbs, camera_radius, camera.position[1])

		# Jump and gravity with ground clamp
		grounded = camera.position[1] <= ground_y + 1e-4
		if grounded:
			camera.position[1] = ground_y
			vertical_velocity = 0.0
			if keys[pygame.K_SPACE]:
				vertical_velocity = jump_velocity
		else:
			vertical_velocity += gravity * dt
		camera.position[1] += vertical_velocity * dt
		if camera.position[1] < ground_y:
			camera.position[1] = ground_y
			vertical_velocity = 0.0

		# Animate one cube; others static
		angle += dt
		animated_cube_world = transform_points(
			cube_verts_base,
			(rotation := (angle * 0.6, angle * 0.9, angle * 0.3)),
			translation=np.array([0.0, cube_half, 4.0], dtype=np.float32),
		)

		# Render
		screen = pygame.display.get_surface()
		renderer.clear(screen, (12, 12, 16))
		# Ground
		renderer.draw_wireframe(screen, camera, grid_vertices, grid_edges, (70, 80, 90))
		# Static cubes
		for verts in static_cube_vertices_world:
			renderer.draw_wireframe(screen, camera, verts, cube_edges, (180, 200, 220))
		# Animated cube
		renderer.draw_wireframe(screen, camera, animated_cube_world, cube_edges, (220, 240, 255))
		# Optional: draw wall as thin box outline (approx via 12 edges)
		# Here we skip rendering wall for simplicity; collision only.

		pygame.display.flip()
		pygame.display.set_caption(
			f"3D Proto - FPS: {clock.get_fps():.1f} | sens {mouse_sensitivity:.3f} speed {move_speed:.1f} smooth {look_smooth:.2f}"
		)

	pygame.mouse.set_visible(True)
	pygame.event.set_grab(False)
	pygame.quit()


def main() -> None:
	parser = argparse.ArgumentParser(description="3D prototype: interactive or headless render")
	parser.add_argument("--headless", action="store_true", help="render frames to output/ without opening a window")
	parser.add_argument("--width", type=int, default=800)
	parser.add_argument("--height", type=int, default=600)
	parser.add_argument("--frames", type=int, default=120, help="frames to render in headless mode")
	# Interactive tuning
	parser.add_argument("--mouse-sensitivity", type=float, default=0.003, help="mouse look sensitivity (radians per pixel)")
	parser.add_argument("--look-smooth", type=float, default=0.5, help="look smoothing strength [0..1], higher = smoother (slower)")
	parser.add_argument("--move-speed", type=float, default=3.0, help="walk speed units/sec")
	parser.add_argument("--sprint-multiplier", type=float, default=1.8, help="sprint speed multiplier when holding Shift")
	parser.add_argument("--camera-radius", type=float, default=0.3, help="camera collision radius")
	parser.add_argument("--gravity", type=float, default=-18.0, help="gravity acceleration for jumping")
	parser.add_argument("--jump-velocity", type=float, default=7.0, help="initial jump velocity")
	args = parser.parse_args()

	if args.headless:
		run_headless(args.width, args.height, args.frames)
	else:
		run_interactive(args.width, args.height, args)


if __name__ == "__main__":
	main()