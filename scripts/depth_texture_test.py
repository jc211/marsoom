import numpy as np
from marsoom.texture import Texture
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pyglet


data = np.load("scripts/depth_data.npy", allow_pickle=True).item()
depth_scale = data["depth_scale"]
color = data["color"]
depth = data["depth"]
depth = depth.astype(np.float32) / 65535.0

t = Texture(width=depth.shape[1], height=depth.shape[0], fmt = pyglet.gl.GL_DEPTH_COMPONENT, internal_format=pyglet.gl.GL_DEPTH_COMPONENT32F)
t.copy_from_host(depth)
# depth[depth > 0.8] = 0.0
# print(depth)
# plt.imshow(depth, vmin=0.0, vmax=0.03)
# plt.show()