#!/usr/bin/env python3.9

from PyBanubaSDK.recognizer import UtilityManager
from PyBanubaSDK.types import FrameData, PixelFormat
from PyBanubaSDK.effect_player import EffectPlayer, EffectPlayerConfiguration, CameraOrientation
from PyBanubaSDK.scene import RenderBackendType
from PyBanubaSDK.utils import EglContext

# # Onscreen rendering
# from bnb.glfw_window import Window

from pathlib import Path

token = 'PLACE YOUR TOKEN HERE'

UtilityManager.initialize(list(__import__('bnb-resources').__path__), token)

from PIL import Image
img = Image.open('face720x1280.png')
pixel_format = PixelFormat.Rgba
data = img.tobytes()

w = img.width
h = img.height

# # Onscreen rendering
# window = Window(w, h)

UtilityManager.load_gl_functions()
EffectPlayer.set_render_backend(RenderBackendType.Opengl)

# Offscreen rendering
context = EglContext.create(w, h)
context.activate()

# help(EffectPlayer)
ep_config = EffectPlayerConfiguration.create(w, h)
ep = EffectPlayer.create(ep_config)
ep.surface_created(w, h)

em = ep.effect_manager()
e = em.load(str(Path('DebugFRX').absolute()))

def render_task():
    fd = FrameData.make_from_bpc8(data, img.width, img.height, CameraOrientation.Deg_0, pixel_format, False, 0, None)
    ep.push_frame_data(fd)
    while ep.draw() < 0:
        pass

def save_image():
    result = ep.read_pixels()
    Image.frombytes("RGBA", (img.width, img.height), result.data.data()).save("result.png")

# Offscreen rendering
render_task()
save_image()
context.deactivate()

# # Onscreen rendering
# window.run_till_closed(render_task)

ep.surface_destroyed()

