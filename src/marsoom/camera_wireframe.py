from typing import Tuple
import numpy as np
import torch
import pyglet
from pyglet.graphics import Batch, Group
from pyglet.math import Mat4
import pyglet.gl as gl

from marsoom.texture import Texture
from marsoom.image_quad import ImageQuad
import marsoom.utils
from marsoom.line_model import LineModel


class CameraWireframe(LineModel):
    def __init__(
        self,
        K: np.ndarray = marsoom.utils.DEFAULT_K,
        width: int = marsoom.utils.DEFAULT_WIDTH,
        height: int = marsoom.utils.DEFAULT_HEIGHT,
        z_offset: float = 0.1,
        frame_color: Tuple[float, float, float, float] = (0.58, 0.58, 0.58, 0.58),
        group: pyglet.graphics.Group = None,
        batch: pyglet.graphics.Batch = None,
    ):
        self.K = K
        self.width = width
        self.height = height
        self.z_offset = z_offset
        self.frame_color = frame_color
        super().__init__(
            self._get_indices(), self._get_vertices(), self._get_colors(), group, batch
        )

    def update_K(self, K: np.ndarray, width: int, height: int):
        if np.allclose(self.K, K):
            return
        self.K = K
        self.width = width
        self.height = height
        self.position = self._get_vertices()
        self._update_vertices()

    def update_z_offset(self, z_offset: float):
        if self.z_offset == z_offset:
            return
        self.z_offset = z_offset
        self.position = self._get_vertices()
        self._update_vertices()

    def update_frame_color(self, frame_color: Tuple[float, float, float, float]):
        if self.frame_color == frame_color:
            return
        self.frame_color = frame_color
        self.colors = self._get_colors()
        self._update_colors()

    def _get_vertices(self):
        K = self.K
        z_offset = self.z_offset
        cx = K[0, 2]
        cy = K[1, 2]
        offset_x = self.width // 2 - cx
        offset_y = self.height // 2 - cy

        top_left = np.array([0.0 + offset_x, 0.0, 1.0], dtype=np.float32)
        top_right = np.array([self.width + offset_x, 0.0, 1.0], dtype=np.float32)
        bot_left = np.array([0.0, self.height + offset_y, 1.0], dtype=np.float32)
        bot_right = np.array(
            [self.width + offset_x, self.height + offset_y, 1.0], dtype=np.float32
        )

        Kinv = np.linalg.inv(K)
        # flip the y and z
        Kinv[1, :] *= -1
        Kinv[2, :] *= -1
        top_left = Kinv @ top_left
        top_right = Kinv @ top_right
        bot_left = Kinv @ bot_left
        bot_right = Kinv @ bot_right

        top_left = top_left / np.linalg.norm(top_left) * z_offset
        top_right = top_right / np.linalg.norm(top_right) * z_offset
        bot_left = bot_left / np.linalg.norm(bot_left) * z_offset
        bot_right = bot_right / np.linalg.norm(bot_right) * z_offset

        width = abs(top_right[0] - top_left[0])
        axis_size = width * 0.1

        self.top_left = top_left.tolist()
        self.top_right = top_right.tolist()
        self.bot_left = bot_left.tolist()
        self.bot_right = bot_right.tolist()

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

        return positions

    def _get_indices(self):
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
        return index

    def _get_colors(self):
        frame_color = self.frame_color
        frame_color = np.array(frame_color, dtype=np.float32)
        red = np.array(marsoom.utils.COLORS["RED"], dtype=np.float32)
        green = np.array(marsoom.utils.COLORS["GREEN"], dtype=np.float32)
        blue = np.array(marsoom.utils.COLORS["BLUE"], dtype=np.float32)
        x_axis_color = np.array([red[0], red[1], red[2], 1.0], dtype=np.float32)
        y_axis_color = np.array([green[0], green[1], green[2], 1.0], dtype=np.float32)
        z_axis_color = np.array([blue[0], blue[1], blue[2], 1.0], dtype=np.float32)
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
        return colors


class CameraWireframeWithImage:
    def __init__(
        self,
        z_offset: float = 0.1,
        width: int = marsoom.utils.DEFAULT_WIDTH,
        height: int = marsoom.utils.DEFAULT_HEIGHT,
        K: np.ndarray = marsoom.utils.DEFAULT_K.copy(),
        frame_color: Tuple[float, float, float, float] = (0.2, 0.2, 0.2, 0.1),
        alpha: float = 1.0,
        texture: Texture | None = None,
        texture_fmt=gl.GL_RGB,
        group: Group | None = None,
        batch: Batch | None = None,
    ):
        if batch is None:
            batch = Batch()

        self.batch = batch

        self.camera_wireframe = CameraWireframe(
            K=K,
            width=width,
            height=height,
            z_offset=z_offset,
            frame_color=frame_color,
            group=group,
            batch=batch,
        )

        if texture:
            self.texture = texture
        else:
            self.texture = Texture(width, height, texture_fmt)

        self.image_quad = ImageQuad(
            tex_id=self.texture.id,
            top_left=self.camera_wireframe.top_left,
            top_right=self.camera_wireframe.top_right,
            bot_right=self.camera_wireframe.bot_right,
            bot_left=self.camera_wireframe.bot_left,
            alpha=alpha,
            group=group,
            batch=batch,
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

    def set_alpha(self, alpha: float):
        self.image_quad.alpha = alpha

    def set_texture_id(self, tex_id: int):
        if self.image_quad.tex_id == tex_id:
            return
        self.image_quad.tex_id = tex_id

    def update_K(self, K: np.ndarray, width: int, height: int):
        self.camera_wireframe.update_K(K, width, height)
        self.image_quad.update(
            top_left=self.camera_wireframe.top_left,
            top_right=self.camera_wireframe.top_right,
            bot_right=self.camera_wireframe.bot_right,
            bot_left=self.camera_wireframe.bot_left,
        )

    def update_z_offset(self, z_offset: float):
        self.camera_wireframe.update_z_offset(z_offset)
        self.image_quad.update(
            top_left=self.camera_wireframe.top_left,
            top_right=self.camera_wireframe.top_right,
            bot_right=self.camera_wireframe.bot_right,
            bot_left=self.camera_wireframe.bot_left,
        )

    def update_frame_color(self, frame_color: Tuple[float, float, float, float]):
        self.camera_wireframe.update_frame_color(frame_color)

    def update_image(self, image: torch.Tensor | np.ndarray):
        if isinstance(image, torch.Tensor):
            self.texture.copy_from_device(image)
        elif isinstance(image, np.ndarray):
            self.texture.copy_from_host(image)
