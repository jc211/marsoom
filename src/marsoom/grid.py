from typing import Tuple
import numpy as np
import torch
import pyglet
from pyglet.graphics import Batch, Group
from pyglet.math import Mat4

from marsoom.line_model import LineModel


class Grid:
    def __init__(self,
                 grid_spacing: float = 0.1,
                 grid_count: int = 10,
                 line_color: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.5),
                 x_axis_color: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0),
                y_axis_color: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
                group: pyglet.graphics.Group = None,
                batch: pyglet.graphics.Batch = None):
        self.grid_spacing = grid_spacing
        self.grid_count = grid_count
        self.line_color = line_color
        self.x_axis_color = x_axis_color
        self.y_axis_color = y_axis_color
        self.grid_axes = GridAxes(grid_spacing, grid_count, x_axis_color, y_axis_color, group, batch)
        self.grid_lines = GridLines(grid_spacing, grid_count, line_color, group, batch)
    
    def update_grid_spacing(self, grid_spacing: float):
        if self.grid_spacing == grid_spacing:
            return
        self.grid_spacing = grid_spacing
        self.grid_lines.update_grid_spacing(grid_spacing)
        self.grid_axes.update_grid_spacing(grid_spacing)
    
    def update_grid_count(self, grid_count: int):
        if self.grid_count == grid_count:
            return
        self.grid_count = grid_count
        self.grid_lines.update_grid_count(grid_count)
        self.grid_axes.update_grid_count(grid_count)


class GridLines(LineModel):
    def __init__(self,
                grid_spacing: float = 0.1,
                grid_count: int = 10,
                line_color: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.5),
                group: pyglet.graphics.Group = None,
                batch: pyglet.graphics.Batch = None):

        self.grid_spacing = grid_spacing
        self.grid_count = grid_count
        self.line_color = line_color
        super().__init__(self._get_indices(), self._get_vertices(), self._get_colors(), group, batch)
    
    def update_grid_spacing(self, grid_spacing: float):
        if self.grid_spacing == grid_spacing:
            return
        self.grid_spacing = grid_spacing
        self.position = self._get_vertices()
        self._update_vertices()
    
    def update_grid_count(self, grid_count: int):
        if self.grid_count == grid_count:
            return
        self.grid_count = grid_count
        self.position = self._get_vertices()
        self._update_vertices()

    
    def _get_vertices(self):
        x_locations = np.linspace(-self.grid_spacing * self.grid_count, self.grid_spacing * self.grid_count, 2 * self.grid_count + 1)
        y_locations = np.linspace(-self.grid_spacing * self.grid_count, self.grid_spacing * self.grid_count, 2 * self.grid_count + 1)   
        points = []
        for i in range(2 * self.grid_count + 1):
            points.extend([x_locations[i], -self.grid_spacing * self.grid_count, 0.0])
            points.extend([x_locations[i], self.grid_spacing * self.grid_count, 0.0])
            points.extend([-self.grid_spacing * self.grid_count, y_locations[i], 0.0])
            points.extend([self.grid_spacing * self.grid_count, y_locations[i], 0.0])
        return np.vstack(points).flatten().astype(np.float32)
    
    def _get_indices(self):
        indices = []
        for i in range(2 * self.grid_count + 1):
            indices.extend([4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3])
        return indices

    
    def _get_colors(self):
        colors = [*self.line_color] * (4 * (2 * self.grid_count + 1))
        return colors


class GridAxes(LineModel):
    def __init__(
        self,
        grid_spacing: float = 0.1,
        grid_count: int = 10,
        x_axis_color: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0),
        y_axis_color: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
        group: pyglet.graphics.Group = None,
        batch: pyglet.graphics.Batch = None,
    ):
        self.grid_spacing = grid_spacing
        self.grid_count = grid_count
        self.x_axis_color = x_axis_color
        self.y_axis_color = y_axis_color
        super().__init__(self._get_indices(), self._get_vertices(), self._get_colors(), group, batch)

    
    @property
    def grid_size(self) -> float:
        return self.grid_spacing * self.grid_count
    
    def update_grid_spacing(self, grid_spacing: float):
        if self.grid_spacing == grid_spacing:
            return
        self.grid_spacing = grid_spacing
        self.position = self._get_vertices()
        self._update_vertices()
    
    def update_grid_count(self, grid_count: int):
        if self.grid_count == grid_count:
            return
        self.grid_count = grid_count
        self.position = self._get_vertices()
        self._update_vertices()
    
    def _get_vertices(self):
        positions = [
            -self.grid_size, 0.0, 0.0, # x-axis
            self.grid_size, 0.0, 0.0, # x-axis
            0.0, -self.grid_size, 0.0, # y-axis
            0.0, self.grid_size, 0.
        ]
        return positions
    
    def _get_indices(self):
        indices = [0, 1, 2, 3]
        return indices
    
    def _get_colors(self):
        # Colors for the grid
        colors = [
            *self.x_axis_color,
            *self.x_axis_color,
            *self.y_axis_color,
            *self.y_axis_color,
        ]
        return colors