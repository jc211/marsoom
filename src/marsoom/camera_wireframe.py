import ctypes
from typing import Tuple
import numpy as np
import torch
import warp as wp

from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet import gl

from embodied_gaussians.vis.texture import Texture
from embodied_gaussians.vis.image_quad import ImageQuad
from embodied_gaussians.utils import COLORS

line_vertex_shader = """
#version 330 core
uniform mat4 world2proj;
uniform mat4 world;
in vec3 position;
in vec4 color;
out vec4 vert_color;
void main() {
    gl_Position = world2proj * world * vec4(position, 1.0);
    vert_color = color;
}
"""

line_fragment_shader = """
#version 330 core
in vec4 vert_color;
out vec4 FragColor;
void main()
{
    FragColor = vert_color; 
}
"""


class CameraWireframeWithImage:
    def __init__(
        self,
        width: int,
        height: int,
        K_opengl: np.ndarray,
        z_offset: float,
        frame_color: Tuple[float, float, float, float] = (0.2, 0.2, 0.2, 0.1),
        X_WV: np.ndarray = np.eye(4, dtype=np.float32),
    ):
        self.camera_wireframe = CameraWireframe(
            K_opengl, z_offset, frame_color=frame_color
        )
        self.texture = Texture(width, height)
        self.image_quad = ImageQuad(self.camera_wireframe.frame_positions)
        self.update_position(X_WV)

    def update_image(self, image: torch.Tensor):
        self.texture.copy_from_device(image)

    def update_position(self, x_wv: np.ndarray):
        self.image_quad.update_position(x_wv)
        self.camera_wireframe.update_position(x_wv)

    def draw(
        self,
        world2proj: np.ndarray,
        model=np.eye(4, dtype=np.float32),
        line_width: float = 1.0,
        alpha: float = 1.0,
    ):
        self.image_quad.draw(world2proj, self.texture.id, alpha=alpha)
        self.camera_wireframe.draw(world2proj, line_width=line_width)


class CameraWireframe:
    def __init__(
        self,
        K_opengl: np.ndarray,
        z_offset,
        frame_color: Tuple[float, float, float, float] = (0.58, 0.58, 0.58, 0.58),
    ):
        top_left = np.array([-1.0, 1.0, z_offset, 1.0], dtype=np.float32)
        top_right = np.array([1.0, 1.0, z_offset, 1.0], dtype=np.float32)
        bot_left = np.array([-1.0, -1.0, z_offset, 1.0], dtype=np.float32)
        bot_right = np.array([1.0, -1.0, z_offset, 1.0], dtype=np.float32)

        Kinv = np.linalg.inv(K_opengl)
        top_left = Kinv @ top_left
        top_right = Kinv @ top_right
        bot_left = Kinv @ bot_left
        bot_right = Kinv @ bot_right

        top_left = top_left[:3] / top_left[3]
        top_right = top_right[:3] / top_right[3]
        bot_left = bot_left[:3] / bot_left[3]
        bot_right = bot_right[:3] / bot_right[3]

        frame_color = np.array(frame_color, dtype=np.float32)
        red = np.array(COLORS["RED"], dtype=np.float32)
        green = np.array(COLORS["GREEN"], dtype=np.float32)
        blue = np.array(COLORS["BLUE"], dtype=np.float32)
        x_axis_color = np.array([red[0], red[1], red[2], 1.0], dtype=np.float32)
        y_axis_color = np.array([green[0], green[1], green[2], 1.0], dtype=np.float32)
        z_axis_color = np.array([blue[0], blue[1], blue[2], 1.0], dtype=np.float32)

        width = abs(top_right[0] - top_left[0])
        axis_size = width * 0.1

        self.frame_positions = [
            *top_left,
            *top_right,
            *bot_right,
            *bot_left,
        ]

        positions = (
            # positions
            0.0,
            0.0,
            0.0,
            *top_left,
            *top_right,
            *bot_right,
            *bot_left,
            # axis
            0.0,
            0.0,
            0.0,
            axis_size,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            axis_size,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            axis_size,
        )

        colors = np.stack(
            [frame_color] * 5
            + [
                x_axis_color,
                x_axis_color,
                y_axis_color,
                y_axis_color,
                z_axis_color,
                z_axis_color,
            ],
        ).flatten()

        index = (
            0,
            1,
            0,
            2,
            0,
            3,
            0,
            4,
            1,
            2,
            2,
            3,
            3,
            4,
            4,
            1,
            # axis
            5,
            6,
            7,
            8,
            9,
            10,
        )
        self.program = ShaderProgram(
            Shader(line_vertex_shader, "vertex"),
            Shader(line_fragment_shader, "fragment"),
        )
        self.vlist = self.program.vertex_list_indexed(
            11, gl.GL_LINES, index, position=("f", positions), color=("f", colors)
        )
        self.x_wv = np.eye(4, dtype=np.float32)

    def update_position(self, x_wv: np.ndarray):
        self.x_wv = x_wv

    def draw(self, world2proj, line_width: float = 1.0):

        self.program.use()
        gl.glLineWidth(line_width)
        self.program["world2proj"] = world2proj
        self.program["world"] = self.x_wv.T.flatten()
        self.vlist.draw(gl.GL_LINES)
        self.program.stop()
