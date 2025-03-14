from pathlib import Path

import numpy as np
import torch
import warp as wp
from pyglet.math import Mat4

import marsoom
from marsoom import imgui, guizmo
import pyglet



# Set device to monitor with opengl context so that the mapping works well
# Especially important if using more than one GPU
device = "cuda:0"
wp.set_device(device)

SCRIPT_PATH = Path(__file__).parent

class CustomWindow(marsoom.Window):
    def __init__(self):
        super().__init__()
        self.viewer = self.create_3D_viewer()
        self.line_renderer = marsoom.LineRenderer()
        self.line_renderer.update(
            torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32),
            torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=torch.float32),
        )

        self.point_renderer = marsoom.PointRenderer()
        self.point_renderer.update(
            torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32),
        )

        self.image_viewer = self.create_2D_viewer(
            "My Image Viewer",
            pixels_to_units=get_pixels_to_meters()
            )
        manip_2d = np.array([[1, 0, 0, 0.0],
                            [0, 1, 0, 0.0],
                            [0, 0, 1, -1.0],
                            [0, 0, 0, 1]], dtype=np.float32)
        self.manip_2d = guizmo.Matrix16(manip_2d.T.flatten())

        self.manip_3d_object = np.eye(4, dtype=np.float32)

        # sphere_mesh = marsoom.renderers.mesh_renderer.create_sphere_mesh()
        # self.example_mesh = marsoom.MeshRenderer(*sphere_mesh)
        self.example_mesh = marsoom.MeshRenderer.from_file(SCRIPT_PATH/"blender_monkey.stl", scale=0.1)
        # self.example_mesh.update(torch.tensor([[0, 0, 0]], dtype=torch.float32).cuda(), torch.tensor([[1, 0, 0, 0]], dtype=torch.float32).cuda(), torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32).cuda(), torch.tensor([[1, 0, 0]], dtype=torch.float32).cuda()) 
        self.example_mesh.update(positions=torch.tensor([[0, 0, 0]], dtype=torch.float32).cuda())


        self.batch = pyglet.graphics.Batch()
        self.example_mesh_2 = pyglet.resource.model("robots/panda/meshes/link0.stl", self.batch)
        self.example_mesh_2.color = (0.0, 1.0, 0.0, 1.0)
        self.example_mesh_2.matrix = Mat4().translate((0.0, 0.0, 1.0))

        self.example_mesh_3 = pyglet.resource.model("robots/panda/meshes/link1.stl", self.batch)
        self.example_mesh_3.color = (1.0, 1.0, 0.0, 1.0)
        self.example_mesh_3.matrix = Mat4().translate((0.0, 0.0, 0.5))

        self.axes_example = marsoom.Axes(batch=self.batch)
        self.axes_example.matrix = Mat4().translate((-1.0, 0.0, 0.0))

        sample_image = torch.randn((480, 640, 3), dtype=torch.float32).to(device)
        sample_image_host = sample_image.cpu().numpy()
        self.image_viewer.update_image(sample_image_host)
        self.camera_1 = marsoom.CameraWireframeWithImage(
            z_offset=0.2,
            batch=self.batch,
            width=640,
            height=480,
        )
        self.camera_1.matrix = Mat4().translate((1.0, 0.0, 0.0))
        self.camera_1.update_image(sample_image_host)
        self.circle = marsoom.Circle(1.0, 1.0, 1.0, 0.3, batch=self.batch)
        pyglet.gl.glPointSize(10)
        self.point = marsoom.Point(0.5, 0.5, 0.0, color=(255, 0, 0), batch=self.batch)


    
    def draw_demo_controls(self):
        imgui.begin("3D Drawing")
        imgui.end()


    def draw(self):
        imgui.begin("3D Drawing")   
        with self.viewer.draw(in_imgui_window=True) as ctx:
            # self.line_renderer.draw(ctx, color=(1, 0, 0))
            # self.point_renderer.draw(ctx, point_size=10)
            # self.example_mesh.draw(ctx)

            self.batch.draw()
            # self.camera_1.draw()

        guizmo.set_id(0)
        self.manip_3d_object = self.viewer.manipulate(
            object_matrix=self.manip_3d_object,
        ) # call after viewer.end so that the image gets drawn
        self.viewer.process_nav()
        imgui.end()

        imgui.begin("2D Viewer")
        self.image_viewer.draw()
        self.image_viewer.axis(unit=marsoom.eViewerUnit.UNIT, scale=0.1)
        self.image_viewer.circle(position=(0, 0), color=(0, 1, 0, 1), radius=100.0, thickness= 10.0)
        self.image_viewer.circle(position=(0.1, 0), color=(0, 1, 1, 1), radius=0.1, thickness= 3.0, unit=marsoom.eViewerUnit.UNIT)
        self.image_viewer.text("Hello World", position=(-10, -10))
        self.image_viewer.polyline(((100, 100), (100, 200), (200, 200), (200, 100)))
        guizmo.set_id(1)
        self.image_viewer.manipulate(object_matrix=self.manip_2d,
            operation=guizmo.OPERATION.translate, 
            mode=guizmo.MODE.local, 
            unit = marsoom.eViewerUnit.UNIT)
        imgui.end()

        # imgui.show_demo_window()

def get_pixels_to_meters():
    pixels_to_meters = np.array(
            [[0.0, 0.001, 0.0], [0.001, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )

    return pixels_to_meters

if __name__ == "__main__":
    window = CustomWindow()
    window.run()