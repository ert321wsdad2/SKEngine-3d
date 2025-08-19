Python 3D Engine (ModernGL + GLFW)

Features
- WASD movement with Shift sprint
- Mouse look (capture with Tab, release with Esc)
- AABB collision against a grid map
- Text map in `assets/map.txt` (1 = wall, 0 = empty)
- Quality presets: F1 low, F2 medium, F3 high

Run
1) Ensure the virtual environment is active:
```
source /workspace/.venv/bin/activate
```
2) Launch:
```
python /workspace/engine/main.py --quality medium --width 1280 --height 720
```

Controls
- WASD: move
- Mouse: look
- Shift: sprint
- Tab: capture mouse
- Esc: release mouse
- F1/F2/F3: quality preset

