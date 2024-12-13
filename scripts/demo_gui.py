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

        self.image_viewer = marsoom.Viewer2D(
            "My Image Viewer",
            pixels_to_units=get_pixels_to_meters()
            )
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
        self.image_viewer.circle(position=(0, 0), color=(0, 1, 0, 1), radius=100.0, thickness= 10.0)
        self.image_viewer.circle(position=(0.1, 0), color=(0, 1, 1, 1), radius=0.1, thickness= 3.0, unit=marsoom.eViewerUnit.UNIT)
        self.image_viewer.text("Hello World", position=(-10, -10))
        self.image_viewer.polyline(((100, 100), (100, 200), (200, 200), (200, 100)))
        imgui.end()


        imgui.show_demo_window()

        imgui.begin("Custom Window!")
        imgui.text("Hello World!")
        imgui.end()

def get_pixels_to_meters():
    side_pixels = 100
    side_meters = 0.1

    p0 = (0, 0)
    p0_map = (0, 0)

    p1 = (side_pixels, side_pixels)
    p1_map = (side_meters, side_meters)

    p2 = (p0[0], p1[1])
    p2_map = (p0_map[0], p1_map[1])
    flip_map = True


    pixels_to_meters = marsoom.compute_affine_transform(
       p0,
       p0_map,
       p1,
       p1_map,
       p2,
       p2_map,
       flip_map
    )
    return pixels_to_meters

if __name__ == "__main__":
    window = CustomWindow()
    window.run()