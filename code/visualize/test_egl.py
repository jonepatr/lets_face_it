import os

os.environ["PYOPENGL_PLATFORM"] = "egl"

import pyrender

pyrender.offscreen.OffscreenRenderer(1, 1)
