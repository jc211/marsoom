from imgui_bundle import imgui
import marsoom
import torch


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
    
    def draw_demo_controls(self):
        imgui.begin("3D Drawing")
        imgui.end()


    def draw(self):
        w2pT = self.viewer.begin("Test")
        self.line_renderer.draw(w2pT, color=(1, 0, 0))
        self.point_renderer.draw(w2pT, point_size=10)

        self.viewer.end()


        imgui.show_demo_window()
        # imgui.begin("Custom Window!")

        # imgui.end()


if __name__ == "__main__":
    window = CustomWindow()
    window.run()