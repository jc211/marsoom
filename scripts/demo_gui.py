import marsoom
from marsoom import imgui, guizmo

import torch
import warp as wp

# Set device to monitor with opengl context so that the mapping works well
# Especially important if using more than one GPU
device = "cuda:0"
wp.set_device(device)
import numpy as np


class CustomWindow(marsoom.Window):
    def __init__(self):
        super().__init__()
        self.viewer = marsoom.Viewer3D()
        self.line_renderer = marsoom.LineRenderer()
        self.line_renderer.update(
            torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32),
            torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=torch.float32),
        )

        self.point_renderer = marsoom.PointRenderer()
        self.point_renderer.update(
            torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32),
        )

        self.image_viewer = marsoom.Viewer2D(
            "My Image Viewer",
            pixels_to_units=get_pixels_to_meters()
            )
        manip_2d = np.array([[1, 0, 0, 0.0],
                            [0, 1, 0, 0.0],
                            [0, 0, 1, -1.0],
                            [0, 0, 0, 1]], dtype=np.float32)
        self.manip_2d = guizmo.Matrix16(manip_2d.T.flatten())

        self.manip_3d_object = guizmo.Matrix16(np.eye(4).flatten())
        # sample_image = torch.randn((480, 640, 3), dtype=torch.float32).to(device)
        # self.image_viewer.update_image(sample_image)
    
    def draw_demo_controls(self):
        imgui.begin("3D Drawing")
        imgui.end()


    def draw(self):
        imgui.begin("3D Drawing")   
        w2pT = self.viewer.begin(in_imgui_window=True)
        self.line_renderer.draw(w2pT, color=(1, 0, 0))
        self.point_renderer.draw(w2pT, point_size=10)
        self.viewer.end()
        guizmo.set_id(0)
        self.viewer.manipulate(
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