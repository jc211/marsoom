from imgui_bundle import imgui
import marsoom
import torch
import warp as wp

# Set device to monitor with opengl context so that the mapping works well
# Especially important if using more than one GPU
device = "cuda:0"
wp.set_device(device)

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

        self.image_viewer = marsoom.Viewer2D("My Image Viewer")
        sample_image = torch.randn((480, 640, 3), dtype=torch.float32).to(device)
        self.image_viewer.update_image(sample_image)
    
    def draw_demo_controls(self):
        imgui.begin("3D Drawing")
        imgui.end()


    def draw(self):
        w2pT = self.viewer.begin("Test")
        self.line_renderer.draw(w2pT, color=(1, 0, 0))
        self.point_renderer.draw(w2pT, point_size=10)

        self.viewer.end()

        imgui.begin("Image Viewer")
        self.image_viewer.draw()
        self.image_viewer.circle("circle0", position_in_pixels=(0, 0), color=(0, 1, 0, 1), radius=100.0, thickness= 10.0)
        self.image_viewer.text("Hello World", position_in_pixels=(-10, -10))
        self.image_viewer.quad("quad1", ((100, 100), (100, 200), (200, 200), (200, 100)))
        imgui.end()


        imgui.show_demo_window()

        imgui.begin("Custom Window!")
        imgui.text("Hello World!")
        imgui.end()


if __name__ == "__main__":
    window = CustomWindow()
    window.run()