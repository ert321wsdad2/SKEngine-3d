#version 330

in vec3 in_position;
in vec3 in_normal;

uniform mat4 u_mvp;
uniform mat4 u_view;

out vec3 v_normal;
out vec3 v_view_normal;

void main() {
	gl_Position = u_mvp * vec4(in_position, 1.0);
	v_normal = in_normal;
	mat3 normal_matrix = mat3(transpose(inverse(u_view)));
	v_view_normal = normalize(normal_matrix * in_normal);
}

