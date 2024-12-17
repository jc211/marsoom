from typing import Tuple
import numpy as np
import pyglet
import marsoom.utils
from marsoom.line_model import LineModel

class Axes(LineModel):
    def __init__(
        self,
        axis_size: float = 0.1,
        group: pyglet.graphics.Group = None,
        batch: pyglet.graphics.Batch = None,
    ):
        index, positions, colors = self._get_vlist_data(axis_size=axis_size)
        super().__init__(index, positions, colors, group, batch)

    def _get_vlist_data(self, axis_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:   
        red = np.array(marsoom.utils.COLORS["RED"], dtype=np.float32)
        green = np.array(marsoom.utils.COLORS["GREEN"], dtype=np.float32)
        blue = np.array(marsoom.utils.COLORS["BLUE"], dtype=np.float32)
        x_axis_color = np.array([red[0], red[1], red[2], 1.0], dtype=np.float32)
        y_axis_color = np.array([green[0], green[1], green[2], 1.0], dtype=np.float32)
        z_axis_color = np.array([blue[0], blue[1], blue[2], 1.0], dtype=np.float32)

        positions = (
            0.0, 0.0, 0.0, # origin
            axis_size, 0.0, 0.0, # x-axis
            0.0, 0.0, 0.0, # origin
            0.0, axis_size, 0.0, # y-axis
            0.0, 0.0, 0.0, # origin
            0.0, 0.0, axis_size, # z-axis
        )

        colors = np.stack(
            [
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
            2,
            3,
            4,
            5,
        ) 
        return index, positions, colors