from typing import Tuple
import numpy as np
import torch
import pyglet
from pyglet.graphics import Batch, Group
from pyglet.math import Mat4

from marsoom.viewer_3d import Context3D 
from marsoom.texture import Texture
from marsoom.image_quad import ImageQuad
import marsoom.utils
from marsoom.line_model import LineModel

class CameraWireframe(LineModel):
    def __init__(
        self,
        K_opengl: np.ndarray = marsoom.utils.DEFAULT_K_OPENGL_T.T,
        z_offset: float = 0.1,
        frame_color: Tuple[float, float, float, float] = (0.58, 0.58, 0.58, 0.58),
        group: pyglet.graphics.Group = None,
        batch: pyglet.graphics.Batch = None,
    ):
        self.K_opengl = K_opengl
        index, positions, colors = self._get_vlist_data(K_opengl, z_offset, frame_color)
        super().__init__(index, positions, colors, group, batch)

    def _get_vlist_data(self, K_opengl: np.ndarray, z_offset: float, frame_color: Tuple[float, float, float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:   
        top_left = np.array([-1.0, 1.0, 0.0, 1.0], dtype=np.float32)
        top_right = np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32)
        bot_left = np.array([-1.0, -1.0, 0.0, 1.0], dtype=np.float32)
        bot_right = np.array([1.0, -1.0, 0.0, 1.0], dtype=np.float32)

        Kinv = np.linalg.inv(K_opengl)
        top_left = Kinv @ top_left
        top_right = Kinv @ top_right
        bot_left = Kinv @ bot_left
        bot_right = Kinv @ bot_right


        top_left = top_left[:3] / top_left[3] 
        top_right = top_right[:3] / top_right[3] 
        bot_left = bot_left[:3] / bot_left[3]  
        bot_right = bot_right[:3] / bot_right[3] 

        top_left = top_left / np.linalg.norm(top_left) * z_offset
        top_right = top_right / np.linalg.norm(top_right) * z_offset
        bot_left = bot_left / np.linalg.norm(bot_left) * z_offset
        bot_right = bot_right / np.linalg.norm(bot_right) * z_offset

        frame_color = np.array(frame_color, dtype=np.float32)
        red = np.array(marsoom.utils.COLORS["RED"], dtype=np.float32)
        green = np.array(marsoom.utils.COLORS["GREEN"], dtype=np.float32)
        blue = np.array(marsoom.utils.COLORS["BLUE"], dtype=np.float32)
        x_axis_color = np.array([red[0], red[1], red[2], 1.0], dtype=np.float32)
        y_axis_color = np.array([green[0], green[1], green[2], 1.0], dtype=np.float32)
        z_axis_color = np.array([blue[0], blue[1], blue[2], 1.0], dtype=np.float32)

        width = abs(top_right[0] - top_left[0])
        axis_size = width * 0.1

        self.top_left = top_left
        self.top_right = top_right
        self.bot_left = bot_left
        self.bot_right = bot_right

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
        return index, positions, colors

class CameraWireframeWithImage:
    def __init__(
        self,
        z_offset: float = 0.1,
        width: int = marsoom.utils.DEFAULT_WIDTH,
        height: int = marsoom.utils.DEFAULT_HEIGHT,
        K_opengl: np.ndarray = marsoom.utils.DEFAULT_K_OPENGL_T.T.copy(),
        frame_color: Tuple[float, float, float, float] = (0.2, 0.2, 0.2, 0.1),
        texture: Texture | None = None,
        group: Group | None = None,
        batch: Batch | None = None,
    ):
        if batch is None:
            batch = Batch()

        self.batch = batch

        self.camera_wireframe = CameraWireframe(
            z_offset=z_offset, K_opengl=K_opengl, frame_color=frame_color,
            group=group, 
            batch=batch
        )

        if texture:
            self.texture = texture
        else:
            self.texture = Texture(width, height)

        self.image_quad = ImageQuad(
            tex_id=self.texture.id,
            top_left=self.camera_wireframe.top_left,
            top_right=self.camera_wireframe.top_right,
            bot_right=self.camera_wireframe.bot_right,
            bot_left=self.camera_wireframe.bot_left,
            group=group,
            batch=batch
        )

    
    @property
    def matrix(self) -> Mat4:
        return self.camera_wireframe.matrix

    @matrix.setter
    def matrix(self, value: Mat4) -> None:
        self.camera_wireframe.matrix = value
        self.image_quad.matrix = value
    
    @property
    def alpha(self) -> float:
        return self.image_quad.alpha
    
    @alpha.setter
    def alpha(self, value: float) -> None:
        self.image_quad.alpha = value

    def update_image(self, image: torch.Tensor):
        self.texture.copy_from_device(image)

    def draw(self):
        self.image_quad.draw()
        self.camera_wireframe.draw()