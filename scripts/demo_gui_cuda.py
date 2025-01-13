from pathlib import Path

import numpy as np
import torch

from pyglet.math import Mat4
import pyglet.gl as gl
import pyglet

import open3d as o3d

import marsoom
from marsoom import imgui, guizmo

import marsoom.grid
import marsoom.texture
import marsoom.cuda

pyglet.options["debug_gl"] = True
pyglet.options["debug_gl_shaders"] = True

SCRIPT_PATH = Path(__file__).parent

class CustomWindow(marsoom.Window):
    def __init__(self):
        super().__init__()
        self.viewer = self.create_3D_viewer()

        self.batch = pyglet.graphics.Batch()
        self.grid = marsoom.grid.Grid(grid_spacing=0.1, grid_count=10, batch=self.batch)


        points = torch.randn(100, 3, dtype=torch.float32)
        colors = torch.rand(100, 4, dtype=torch.float32)
        self.points = marsoom.cuda.PointRenderer()
        self.mesh_renderer = marsoom.cuda.InstancedMeshRenderer.from_open3d_mesh(
            mesh=o3d.geometry.TriangleMesh.create_sphere(radius=0.1),
        )
        positions = torch.rand(10, 3, dtype=torch.float32).cuda()
        positions[:,0] -= 1.0
        self.mesh_renderer.update(
            positions=positions,
        )

        self.ellipse_renderer = marsoom.cuda.EllipseRenderer()
        self.ellipse_mesh_renderer = marsoom.cuda.InstancedMeshRenderer.from_open3d_mesh(
            mesh=o3d.geometry.TriangleMesh.create_sphere(radius=0.1),
        )
        num_ellipses = 100
        positions = torch.rand(num_ellipses, 3, dtype=torch.float32).cuda()
        positions[:, 0] += -1.0
        quats = torch.zeros(num_ellipses, 4, dtype=torch.float32).cuda()
        quats[:, 0] = 1.0
        scales = torch.rand(num_ellipses, 3, dtype=torch.float32).cuda()* 0.2

        colors = torch.rand(num_ellipses, 3, dtype=torch.float32).cuda()
        conics = torch.zeros(num_ellipses, 3, dtype=torch.float32).cuda()
        conics[:, 0] = 1/100.0
        conics[:, 2] = 1/100.0
        self.ellipse_mesh_renderer.update(
            colors=colors,
            positions=positions,
            rotations=quats,
            scaling=scales
        )
        # line size
        self.ellipse_renderer.update(
            positions=positions,
            colors=colors,
            conics=conics,
            opacity=torch.rand(num_ellipses, 1, dtype=torch.float32).cuda()
        )



    def draw_demo_controls(self):
        imgui.begin("Debug")
        _, self.viewer.orthogonal = imgui.checkbox("Orthogonal", self.viewer.orthogonal)
        _, self.viewer.fl_x = imgui.input_float("Focal Length X", self.viewer.fl_x)
        _, self.viewer.fl_y = imgui.input_float("Focal Length Y", self.viewer.fl_y)
        if imgui.button("Top View"):
            self.viewer.top_view()
        if imgui.button("Front View"):
            self.viewer.front_view()
        if imgui.button("Side View"):
            self.viewer.right_view()
        if imgui.button("Reset View"):
            self.viewer.reset_view()

        imgui.end()


    def render(self):
        self.draw_demo_controls()

        imgui.begin("3D Drawing")   
        pyglet.gl.glPointSize(6)
        self.points.update_positions_and_colors(
            torch.rand(10000, 3, dtype=torch.float32).cuda(),
            torch.rand(10000, 4, dtype=torch.float32).cuda()
        )
        with self.viewer.draw(in_imgui_window=True) as ctx:
            self.batch.draw()
            # self.points.draw()
            # self.mesh_renderer.draw()
            self.ellipse_mesh_renderer.draw()
            self.ellipse_renderer.draw(5.0)
            gl.glLineWidth(1.0)

        guizmo.set_id(0)
        self.viewer.process_nav()
        imgui.end()



def get_pixels_to_meters():
    pixels_to_meters = np.array(
            [[0.0, 0.001, 0.0], [0.001, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )

    return pixels_to_meters

if __name__ == "__main__":
    window = CustomWindow()
    window.run()