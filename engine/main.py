from dataclasses import dataclass


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