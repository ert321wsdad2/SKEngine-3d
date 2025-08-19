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