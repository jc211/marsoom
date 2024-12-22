from pathlib import Path

import numpy as np
from pyglet.math import Mat4

import marsoom
from marsoom import imgui, guizmo
import pyglet


SCRIPT_PATH = Path(__file__).parent

class CustomWindow(marsoom.Window):
    def __init__(self):
        super().__init__()
        self.viewer = self.create_3D_viewer()

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


        self.batch = pyglet.graphics.Batch()
        self.example_mesh_2 = pyglet.resource.model("robots/panda/meshes/link0.stl", self.batch)
        self.example_mesh_2.color = (0.0, 1.0, 0.0, 1.0)
        self.example_mesh_2.matrix = Mat4().translate((0.0, 0.0, 1.0))

        self.example_mesh_3 = pyglet.resource.model("robots/panda/meshes/link1.stl", self.batch)
        self.example_mesh_3.color = (1.0, 1.0, 0.0, 1.0)
        self.example_mesh_3.matrix = Mat4().translate((0.0, 0.0, 0.5))

        self.axes_example = marsoom.Axes(batch=self.batch)
        self.axes_example.matrix = Mat4().translate((-1.0, 0.0, 0.0))

        sample_image = np.random.randn(480, 640, 3).astype(np.float32)
        self.image_viewer.update_image(sample_image)
        self.camera_1 = marsoom.CameraWireframeWithImage(
            z_offset=0.2,
            batch=self.batch,
            width=640,
            height=480,
        )
        self.camera_1.matrix = Mat4().translate((1.0, 0.0, 0.0))
        self.camera_1.update_image(sample_image)
        self.circle = marsoom.Circle(1.0, 1.0, 1.0, 0.3, batch=self.batch)
        pyglet.gl.glPointSize(10)
        self.point = marsoom.Point(0.5, 0.5, 0.0, color=(255, 0, 0), batch=self.batch)


    
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


    def draw(self):
        self.draw_demo_controls()

        imgui.begin("3D Drawing")   
        with self.viewer.draw(in_imgui_window=True) as ctx:
            self.batch.draw()

        guizmo.set_id(0)
        self.manip_3d_object = self.viewer.manipulate(
            object_matrix=self.manip_3d_object,
        ) 
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


def get_pixels_to_meters():
    pixels_to_meters = np.array(
            [[0.0, 0.001, 0.0], [0.001, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )

    return pixels_to_meters

if __name__ == "__main__":
    window = CustomWindow()
    window.run()