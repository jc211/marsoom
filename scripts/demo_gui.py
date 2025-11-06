from pathlib import Path

import numpy as np
import pyglet
from pyglet.math import Mat4

import marsoom
import marsoom.grid
import marsoom.texture
from marsoom import guizmo, imgui

SCRIPT_PATH = Path(__file__).parent


class CustomWindow(marsoom.Window):
    def __init__(self):
        super().__init__()
        self.viewer = self.create_3D_viewer()

        self.image_viewer = self.create_2D_viewer(
            "My Image Viewer", pixels_to_units=get_pixels_to_meters()
        )
        manip_2d = np.array(
            [[1, 0, 0, 0.0], [0, 1, 0, 0.0], [0, 0, 1, -1.0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        self.manip_2d = guizmo.Matrix16(manip_2d.T.flatten())

        self.manip_3d_object = np.eye(4, dtype=np.float32)

        self.batch = pyglet.graphics.Batch()
        self.grid = marsoom.grid.Grid(grid_spacing=0.1, grid_count=10, batch=self.batch)

        points = np.random.randn(100, 3).astype(np.float32)
        colors = np.random.rand(100, 3).astype(np.float32)
        self.points = marsoom.Points(points=points, colors=colors, batch=self.batch)

        self.example_mesh_2 = pyglet.resource.model(
            "robots/panda/meshes/link0.stl", self.batch
        )
        self.example_mesh_2.color = (0.0, 1.0, 0.0, 1.0)
        self.example_mesh_2.matrix = Mat4().translate((0.0, 0.0, 1.0))

        self.example_mesh_3 = pyglet.resource.model(
            "robots/panda/meshes/link1.stl", self.batch
        )
        self.example_mesh_3.color = (1.0, 1.0, 0.0, 1.0)
        self.example_mesh_3.matrix = Mat4().translate((0.0, 0.0, 0.5))

        self.axes_example = marsoom.Axes(batch=self.batch)
        self.axes_example.matrix = Mat4().translate((-1.0, 0.0, 0.0))

        sample_image = np.random.randn(480, 640, 3).astype(np.float32)
        self.image_viewer.update_image(sample_image)
        self.camera_batch = pyglet.graphics.Batch()
        self.camera_1 = marsoom.CameraWireframeWithImage(
            batch=self.camera_batch,
            width=640,
            height=480,
        )
        # self.camera_1.matrix = Mat4().translate((1.0, 0.0, 0.0))
        self.camera_1.update_image(sample_image)
        self.circle = marsoom.Circle(1.0, 1.0, 1.0, 0.3, batch=self.batch)
        self.point = marsoom.Point(0.5, 0.5, 0.0, color=(255, 0, 0), batch=self.batch)

        data = np.load("scripts/depth_data.npy", allow_pickle=True).item()
        depth = data["depth"]
        color = data["color"]
        depth_scale = data["depth_scale"]
        K = data["K"]

        # pointcloud
        # convert depth map to pointcloud
        # use open3d
        # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        #     o3d.geometry.Image(color),
        #     o3d.geometry.Image(depth),
        #     depth_scale=1.0/depth_scale,
        #     depth_trunc=10.0,
        #     convert_rgb_to_intensity=False
        # )
        # pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        #     rgbd_image,
        #     o3d.camera.PinholeCameraIntrinsic(
        #         width=color.shape[1],
        #         height=color.shape[0],
        #         fx=K[0, 0],
        #         fy=K[1, 1],
        #         cx=K[0, 2],
        #         cy=K[1, 2]
        #     )
        # )

        # # visualize
        # # o3d.visualization.draw_geometries([pcd])

        # points = np.asarray(pcd.points)
        # # flip z and y
        # points[:, 1] = -points[:, 1]
        # points[:, 2] = -points[:, 2]
        # colors = np.asarray(pcd.colors)
        # # self.points_3d = marsoom.Points(points=points, colors=colors, batch=self.batch_sc)

        # self.batch_sc = pyglet.graphics.Batch()

        self.camera_1.update_K(K, width=color.shape[1], height=color.shape[0])

        width = color.shape[1]
        height = color.shape[0]
        self.sc = marsoom.StructuredPointCloud(width, height)
        self.sc.update_intrinsics(K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        # self.camera_1.update_image((color/255.0).astype(np.float32))
        self.camera_1.update_image(color)
        self.tex_color = marsoom.texture.Texture(
            width=width,
            height=height,
            fmt=pyglet.gl.GL_BGR,
            internal_format=pyglet.gl.GL_RGBA,
        )
        # self.tex_color.copy_from_host((color/255.0).astype(np.float32))
        self.tex_color.copy_from_host(color)
        # depth[depth == 65535] = 0
        # depth = depth.astype(np.float32)*depth_scale
        self.sc.depth_scale = depth_scale
        self.depth = depth
        self.sc.update_depth(self.depth)
        self.sc.color_texture_id = self.tex_color.tex.id

        self.draw_overlay = False
        self.overlay_tex = marsoom.texture.Texture(
            width=width,
            height=height,
            fmt=pyglet.gl.GL_RGB,
            internal_format=pyglet.gl.GL_RGBA,
        )
        image = np.random.rand(height, width, 3).astype(np.float32)
        self.overlay_tex.copy_from_host(image)

        self.overlay = marsoom.Overlay(self.overlay_tex.id, alpha=0.5)

    def draw_demo_controls(self):
        imgui.begin("Debug")
        _, self.draw_overlay = imgui.checkbox("Draw Overlay", self.draw_overlay)
        c, new_alpha = imgui.slider_float("Overlay Alpha", self.overlay.alpha, 0.0, 1.0)
        if c:
            self.overlay.alpha = new_alpha

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

        # self.depth += np.random.randn(*self.depth.shape).astype(np.float32)*0.001
        self.sc.update_depth(self.depth)

        with self.viewer.draw(in_imgui_window=True) as ctx:
            # self.batch.draw()
            self.camera_batch.draw()
            self.sc.draw()
            if self.draw_overlay:
                self.overlay.draw()

        guizmo.push_id(0)
        changed, self.manip_3d_object = self.viewer.manipulate(
            object_matrix=self.manip_3d_object
        )
        guizmo.pop_id()
        self.viewer.process_nav()
        imgui.end()

        imgui.begin("2D Viewer")
        self.image_viewer.draw()
        self.image_viewer.axis(unit=marsoom.eViewerUnit.UNIT, scale=0.1)
        self.image_viewer.circle(
            position=(0, 0), color=(0, 1, 0, 1), radius=100.0, thickness=10.0
        )
        self.image_viewer.circle(
            position=(0.1, 0),
            color=(0, 1, 1, 1),
            radius=0.1,
            thickness=3.0,
            unit=marsoom.eViewerUnit.UNIT,
        )
        self.image_viewer.text("Hello World", position=(-10, -10))
        self.image_viewer.polyline(((100, 100), (100, 200), (200, 200), (200, 100)))
        guizmo.push_id(1)
        self.image_viewer.manipulate(
            object_matrix=self.manip_2d,
            operation=guizmo.OPERATION.translate,
            mode=guizmo.MODE.local,
            unit=marsoom.eViewerUnit.UNIT,
        )
        guizmo.pop_id()
        imgui.end()


def get_pixels_to_meters():
    pixels_to_meters = np.array([[0.0, 0.001, 0.0], [0.001, 0.0, 0.0], [0.0, 0.0, 1.0]])

    return pixels_to_meters


if __name__ == "__main__":
    window = CustomWindow()
    window.run()
