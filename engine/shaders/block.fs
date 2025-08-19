#version 330

in vec3 v_normal;
in vec3 v_view_normal;

uniform vec3 u_color;
uniform vec3 u_light_dir;

out vec4 fragColor;

void main() {
	vec3 n = normalize(v_normal);
	float ndl = max(dot(n, normalize(u_light_dir)), 0.0);
	float ambient = 0.25;
	float diffuse = ndl * 0.7;
	float lighting = clamp(ambient + diffuse, 0.0, 1.0);
	vec3 color = u_color * lighting;
	fragColor = vec4(color, 1.0);
}

