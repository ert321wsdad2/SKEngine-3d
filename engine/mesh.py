import numpy as np


def create_cube_mesh(size: float = 1.0):
	h = size * 0.5
	vertices = np.array(
		[
			[-h, -h, -h],
			[+h, -h, -h],
			[+h, +h, -h],
			[-h, +h, -h],
			[-h, -h, +h],
			[+h, -h, +h],
			[+h, +h, +h],
			[-h, +h, +h],
		],
		dtype=np.float32,
	)

	edges = [
		(0, 1), (1, 2), (2, 3), (3, 0),
		(4, 5), (5, 6), (6, 7), (7, 4),
		(0, 4), (1, 5), (2, 6), (3, 7),
	]

	return vertices, edges


def create_grid_mesh(width: int = 50, depth: int = 50, step: float = 1.0, y: float = 0.0):
	"""Create a grid (XZ plane) as a wireframe mesh at height y."""
	lines = []
	w_half = width * 0.5 * step
	d_half = depth * 0.5 * step

	# Lines parallel to X (varying Z)
	for i in range(depth + 1):
		z = -d_half + i * step
		lines.append(([-w_half, y, z], [w_half, y, z]))

	# Lines parallel to Z (varying X)
	for i in range(width + 1):
		x = -w_half + i * step
		lines.append(([x, y, -d_half], [x, y, d_half]))

	# Convert to vertices/edges
	vertices = []
	edges = []
	for idx, (p0, p1) in enumerate(lines):
		v_start = len(vertices)
		vertices.extend([p0, p1])
		edges.append((v_start + 0, v_start + 1))

	return np.array(vertices, dtype=np.float32), edges