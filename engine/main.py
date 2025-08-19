import os
import math
import time
import argparse
from dataclasses import dataclass

import glfw
import moderngl
import numpy as np
from pyrr import Matrix44, Vector3


WINDOW_TITLE = "Python 3D Engine (WASD + Mouse Look + Collisions)"


def resource_path(*parts: str) -> str:
	return os.path.join(os.path.dirname(__file__), *parts)


@dataclass
class QualityPreset:
	name: str
	render_distance: int
	vsync: int
	face_cull: bool
	wireframe: bool
	msaa_samples: int


QUALITY_PRESETS = {
	"low": QualityPreset(name="low", render_distance=24, vsync=0, face_cull=True, wireframe=False, msaa_samples=0),
	"medium": QualityPreset(name="medium", render_distance=48, vsync=1, face_cull=True, wireframe=False, msaa_samples=2),
	"high": QualityPreset(name="high", render_distance=96, vsync=1, face_cull=True, wireframe=False, msaa_samples=4),
}


class FPSCamera:
	def __init__(self, position: Vector3):
		self.position = Vector3(position)
		self.yaw = 90.0
		self.pitch = 0.0
		self.mouse_sensitivity = 0.08
		self.speed = 4.0

	def get_forward(self) -> Vector3:
		cy = math.cos(math.radians(self.yaw))
		sy = math.sin(math.radians(self.yaw))
		cp = math.cos(math.radians(self.pitch))
		sp = math.sin(math.radians(self.pitch))
		return Vector3([cy * cp, sp, sy * cp])

	def get_right(self) -> Vector3:
		fwd = self.get_forward()
		fwd[1] = 0.0
		if np.linalg.norm(fwd) > 0:
			fwd = fwd / np.linalg.norm(fwd)
		right = Vector3([fwd.z, 0.0, -fwd.x])
		return right

	def get_view_matrix(self) -> Matrix44:
		forward = self.get_forward()
		target = self.position + forward
		up = Vector3([0.0, 1.0, 0.0])
		return Matrix44.look_at(self.position, target, up)


class InputState:
	def __init__(self, window):
		self.window = window
		self.first_mouse = True
		self.last_cursor_x = 0.0
		self.last_cursor_y = 0.0
		self.mouse_captured = False

	def capture_mouse(self, capture: bool):
		self.mouse_captured = capture
		mode = glfw.CURSOR_DISABLED if capture else glfw.CURSOR_NORMAL
		glfw.set_input_mode(self.window, glfw.CURSOR, mode)


class Map:
	def __init__(self, grid: list[list[int]]):
		self.grid = grid
		self.height = len(grid)
		self.width = len(grid[0]) if self.height > 0 else 0

	@staticmethod
	def from_file(path: str) -> "Map":
		if not os.path.exists(path):
			raise FileNotFoundError(f"Map file not found: {path}")
		grid: list[list[int]] = []
		with open(path, "r", encoding="utf-8") as f:
			for raw in f.readlines():
				line = raw.strip()
				if not line:
					continue
				row = []
				for ch in line:
					row.append(1 if ch != '0' else 0)
				grid.append(row)
		maxw = max((len(r) for r in grid), default=0)
		for r in grid:
			if len(r) < maxw:
				r.extend([0] * (maxw - len(r)))
		return Map(grid)

	def is_solid(self, x: int, z: int) -> bool:
		if x < 0 or z < 0 or z >= self.height or x >= self.width:
			return True
		return self.grid[z][x] != 0


class Mesh:
	def __init__(self, ctx: moderngl.Context, vertices: np.ndarray):
		self.ctx = ctx
		self.vbo = ctx.buffer(vertices.tobytes())
		self.vao = None
		self.vertex_count = len(vertices)

	def build_vao(self, program: moderngl.Program):
		self.vao = self.ctx.vertex_array(
			program,
			[(self.vbo, "3f 3f", "in_position", "in_normal")],
		)

	def render(self):
		if self.vao is not None:
			self.vao.render(mode=moderngl.TRIANGLES)


def build_world_mesh(map_data: Map) -> np.ndarray:
	vertices: list[float] = []

	def add_face(px, py, pz, normal, corners):
		for c in corners:
			vertices.extend([px + c[0], py + c[1], pz + c[2], normal[0], normal[1], normal[2]])

	faces = {
		"px": (Vector3([1, 0, 0]), [[1,0,0],[1,1,0],[1,1,1], [1,0,0],[1,1,1],[1,0,1]]),
		"nx": (Vector3([-1,0,0]), [[0,0,0],[0,1,1],[0,1,0], [0,0,0],[0,0,1],[0,1,1]]),
		"py": (Vector3([0, 1, 0]), [[0,1,0],[1,1,0],[1,1,1], [0,1,0],[1,1,1],[0,1,1]]),
		"ny": (Vector3([0,-1, 0]), [[0,0,0],[1,0,1],[1,0,0], [0,0,0],[0,0,1],[1,0,1]]),
		"pz": (Vector3([0, 0, 1]), [[0,0,1],[1,1,1],[1,0,1], [0,0,1],[0,1,1],[1,1,1]]),
		"nz": (Vector3([0, 0,-1]), [[0,0,0],[1,0,0],[1,1,0], [0,0,0],[1,1,0],[0,1,0]]),
	}

	for z in range(map_data.height):
		for x in range(map_data.width):
			if map_data.grid[z][x] == 0:
				continue
			x0 = float(x)
			z0 = float(z)
			if not map_data.is_solid(x + 1, z):
				add_face(x0, 0.0, z0, faces["px"][0], faces["px"][1])
			if not map_data.is_solid(x - 1, z):
				add_face(x0, 0.0, z0, faces["nx"][0], faces["nx"][1])
			if not map_data.is_solid(x, z + 1):
				add_face(x0, 0.0, z0, faces["pz"][0], faces["pz"][1])
			if not map_data.is_solid(x, z - 1):
				add_face(x0, 0.0, z0, faces["nz"][0], faces["nz"][1])
			add_face(x0, 1.0, z0, faces["py"][0], faces["py"][1])
			add_face(x0, 0.0, z0, faces["ny"][0], faces["ny"][1])

	if not vertices:
		return np.zeros((0, 6), dtype=np.float32)

	return np.array(vertices, dtype=np.float32).reshape(-1, 6)


def resolve_move_with_collision(map_data: Map, start_pos: Vector3, proposed_pos: Vector3, half_extents: Vector3) -> Vector3:
	resolved = Vector3(start_pos)
	hx, hz = float(half_extents.x), float(half_extents.z)

	def overlaps_solid(x: float, z: float) -> bool:
		min_x = int(math.floor(x - hx))
		max_x = int(math.floor(x + hx))
		min_z = int(math.floor(z - hz))
		max_z = int(math.floor(z + hz))
		for tz in range(min_z, max_z + 1):
			for tx in range(min_x, max_x + 1):
				if map_data.is_solid(tx, tz):
					return True
		return False

	target_x = float(proposed_pos.x)
	dir_x = 0.0 if target_x == resolved.x else (1.0 if target_x > resolved.x else -1.0)
	if overlaps_solid(target_x, resolved.z):
		cell_edge = math.floor(target_x + hx * dir_x)
		resolved.x = cell_edge - hx * dir_x
	else:
		resolved.x = target_x

	target_z = float(proposed_pos.z)
	dir_z = 0.0 if target_z == resolved.z else (1.0 if target_z > resolved.z else -1.0)
	if overlaps_solid(resolved.x, target_z):
		cell_edge = math.floor(target_z + hz * dir_z)
		resolved.z = cell_edge - hz * dir_z
	else:
		resolved.z = target_z

	resolved.y = start_pos.y
	return resolved


class Engine:
	def __init__(self, quality: QualityPreset, width: int = 1280, height: int = 720):
		if not glfw.init():
			raise RuntimeError("Failed to initialize GLFW")

		glfw.window_hint(glfw.SAMPLES, quality.msaa_samples)
		self.window = glfw.create_window(width, height, WINDOW_TITLE, None, None)
		if not self.window:
			glfw.terminate()
			raise RuntimeError("Failed to create GLFW window")

		glfw.make_context_current(self.window)
		glfw.swap_interval(quality.vsync)

		self.ctx = moderngl.create_context()
		self.ctx.enable(moderngl.DEPTH_TEST)
		if quality.face_cull:
			self.ctx.enable(moderngl.CULL_FACE)
		self.ctx.wireframe = quality.wireframe
		if quality.msaa_samples > 0:
			self.ctx.enable(moderngl.MULTISAMPLE)
		else:
			self.ctx.disable(moderngl.MULTISAMPLE)

		self.input = InputState(self.window)
		self.input.capture_mouse(True)
		glfw.set_cursor_pos_callback(self.window, self.on_mouse_move)
		glfw.set_key_callback(self.window, self.on_key)

		self.camera = FPSCamera(position=Vector3([2.5, 1.6, 2.5]))
		self.player_half_extents = Vector3([0.3, 0.9, 0.3])

		with open(resource_path("shaders", "block.vs"), "r", encoding="utf-8") as f:
			vs_src = f.read()
		with open(resource_path("shaders", "block.fs"), "r", encoding="utf-8") as f:
			fs_src = f.read()
		self.program = self.ctx.program(vertex_shader=vs_src, fragment_shader=fs_src)

		map_path = resource_path("assets", "map.txt")
		self.map = Map.from_file(map_path)
		vertices = build_world_mesh(self.map)
		self.mesh = Mesh(self.ctx, vertices)
		self.mesh.build_vao(self.program)

		self.aspect = width / height
		self.projection = Matrix44.perspective_projection(75.0, self.aspect, 0.1, float(quality.render_distance))

		self.program["u_color"].value = (0.75, 0.85, 0.95)
		self.program["u_light_dir"].value = tuple(Vector3([0.5, 1.0, 0.2]).normalized)

		self.last_time = time.time()
		self.quality = quality

	def on_mouse_move(self, window, xpos, ypos):
		if not self.input.mouse_captured:
			return
		if self.input.first_mouse:
			self.input.last_cursor_x = xpos
			self.input.last_cursor_y = ypos
			self.input.first_mouse = False
			return
		dx = float(xpos - self.input.last_cursor_x)
		dy = float(ypos - self.input.last_cursor_y)
		self.input.last_cursor_x = xpos
		self.input.last_cursor_y = ypos

		self.camera.yaw += dx * self.camera.mouse_sensitivity
		self.camera.pitch -= dy * self.camera.mouse_sensitivity
		self.camera.pitch = max(-89.0, min(89.0, self.camera.pitch))

	def on_key(self, window, key, scancode, action, mods):
		if action == glfw.PRESS:
			if key == glfw.KEY_ESCAPE:
				self.input.capture_mouse(False)
			if key == glfw.KEY_F1:
				self.set_quality("low")
			if key == glfw.KEY_F2:
				self.set_quality("medium")
			if key == glfw.KEY_F3:
				self.set_quality("high")
		elif action == glfw.RELEASE:
			if key == glfw.KEY_TAB:
				self.input.capture_mouse(True)

	def set_quality(self, name: str):
		q = QUALITY_PRESETS.get(name, self.quality)
		self.quality = q
		glfw.swap_interval(q.vsync)
		if q.face_cull:
			self.ctx.enable(moderngl.CULL_FACE)
		else:
			self.ctx.disable(moderngl.CULL_FACE)
		self.ctx.wireframe = q.wireframe
		self.projection = Matrix44.perspective_projection(75.0, self.aspect, 0.1, float(q.render_distance))
		if q.msaa_samples > 0:
			self.ctx.enable(moderngl.MULTISAMPLE)
		else:
			self.ctx.disable(moderngl.MULTISAMPLE)

	def poll_movement(self, dt: float):
		speed = self.camera.speed
		if glfw.get_key(self.window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS:
			speed *= 1.8
		move_dir = Vector3([0.0, 0.0, 0.0])
		if glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS:
			move_dir += self.camera.get_forward()
		if glfw.get_key(self.window, glfw.KEY_S) == glfw.PRESS:
			move_dir -= self.camera.get_forward()
		if glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS:
			move_dir -= self.camera.get_right()
		if glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS:
			move_dir += self.camera.get_right()

		move_dir[1] = 0.0
		norm = np.linalg.norm(move_dir)
		if norm > 0:
			move_dir = move_dir / norm
			delta = move_dir * (speed * dt)
			proposed = self.camera.position + delta
			resolved = resolve_move_with_collision(self.map, self.camera.position, proposed, self.player_half_extents)
			self.camera.position = resolved

	def render(self):
		self.ctx.clear(0.05, 0.06, 0.08, 1.0)
		view = self.camera.get_view_matrix()
		self.program["u_mvp"].write((self.projection * view).astype("f4").tobytes())
		self.program["u_view"].write(view.astype("f4").tobytes())
		self.mesh.render()

	def run(self):
		while not glfw.window_should_close(self.window):
			now = time.time()
			dt = float(now - self.last_time)
			self.last_time = now
			glfw.poll_events()
			self.poll_movement(dt)
			self.render()
			glfw.swap_buffers(self.window)

		glfw.terminate()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Simple Python 3D engine with WASD, mouse look, collisions, and quality presets.")
	parser.add_argument("--quality", choices=list(QUALITY_PRESETS.keys()), default="medium", help="Quality preset")
	parser.add_argument("--width", type=int, default=1280)
	parser.add_argument("--height", type=int, default=720)
	return parser.parse_args()


def main():
	args = parse_args()
	engine = Engine(quality=QUALITY_PRESETS[args.quality], width=args.width, height=args.height)
	engine.run()


if __name__ == "__main__":
	main()
