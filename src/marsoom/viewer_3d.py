from typing import Optional, Tuple
import sys
import ctypes
import math

import numpy as np
import torch

from pyglet import gl
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.math import Vec3 as PyVec3
from pyglet.math import Mat4 as PyMat4
from imgui_bundle import imgui, ImVec2

from marsoom.renderers.axis_renderer import AxisRenderer


frame_vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

void main() {
    gl_Position = vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}
"""

frame_fragment_shader = """
#version 330 core
in vec2 TexCoord;

out vec4 FragColor;

uniform sampler2D textureSampler;

void main() {
    FragColor = texture(textureSampler, TexCoord);
}
"""

frame_depth_fragment_shader = """
#version 330 core
in vec2 TexCoord;

out vec4 FragColor;

uniform sampler2D textureSampler;

vec3 bourkeColorMap(float v) {
    vec3 c = vec3(1.0, 1.0, 1.0);

    v = clamp(v, 0.0, 1.0); // Ensures v is between 0 and 1

    if (v < 0.25) {
        c.r = 0.0;
        c.g = 4.0 * v;
    } else if (v < 0.5) {
        c.r = 0.0;
        c.b = 1.0 + 4.0 * (0.25 - v);
    } else if (v < 0.75) {
        c.r = 4.0 * (v - 0.5);
        c.b = 0.0;
    } else {
        c.g = 1.0 + 4.0 * (0.75 - v);
        c.b = 0.0;
    }

    return c;
}

void main() {
    float depth = texture(textureSampler, TexCoord).r;
    FragColor = vec4(bourkeColorMap(sqrt(1.0 - depth)), 1.0);
}
"""

def str_buffer(string: str):
    return ctypes.c_char_p(string.encode("utf-8"))

def convert_K_to_projection_matrixT(
    K: np.ndarray, width: int, height: int, near: float = 0.0001, far: float = 10.0
) -> np.ndarray:
    fx, fy, u0, v0 = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    l = -u0 * near / fx
    r = (width - u0) * near / fx
    b = -(height - v0) * near / fy
    t = v0 * near / fy

    return np.array(
        [
            [2 * near / (r - l), 0, (r + l) / (r - l), 0],
            [0, 2 * near / (t - b), (t + b) / (t - b), 0],
            [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
            [0, 0, -1.0, 0],
        ],
        dtype=np.float32,
    ).T

def calibration_matrix_values(camera_matrix: np.ndarray, image_size: Tuple[int, int], aperture_width: float=0.0, aperture_height: float=0.0):
    # https://github.com/opencv/opencv/blob/93b607dc72e1d7953b17a58d8fb5f130b05c3d7a/modules/calib3d/src/calibration.cpp#L3988
    if camera_matrix.shape != (3, 3):
        raise ValueError("Size of camera_matrix must be 3x3!")
    
    K = camera_matrix
    assert image_size[0] != 0 and image_size[1] != 0 and K[0, 0] != 0.0 and K[1, 1] != 0.0

    # Calculate pixel aspect ratio
    aspect_ratio = K[1, 1] / K[0, 0]

    # Calculate number of pixel per realworld unit
    if aperture_width != 0.0 and aperture_height != 0.0:
        mx = image_size[0] / aperture_width
        my = image_size[1] / aperture_height
    else:
        mx = 1.0
        my = aspect_ratio

    # Calculate fovx and fovy
    fovx = math.atan2(K[0, 2], K[0, 0]) + math.atan2(image_size[0] - K[0, 2], K[0, 0])
    fovy = math.atan2(K[1, 2], K[1, 1]) + math.atan2(image_size[1] - K[1, 2], K[1, 1])
    fovx = fovx * 180.0 / math.pi
    fovy = fovy * 180.0 / math.pi

    # Calculate focal length
    focal_length = K[0, 0] / mx

    # Calculate principle point
    principal_point = (K[0, 2] / mx, K[1, 2] / my)

    return {
        'fovx': fovx,
        'fovy': fovy,
        'focal_length': focal_length,
        'principal_point': principal_point,
        'aspect_ratio': aspect_ratio
    }

class Viewer3D:
    screen_width: int = 0
    screen_height: int = 0
    fl_x: float = 500.0
    fl_y: float = 500.0
    screen_center_x: float = 0.5
    screen_center_y: float = 0.5
    background_color: Tuple[float, float, float] = (0.86, 0.86, 0.86)
    camera_speed: float = 0.01
    point_size = 3.0
    line_width = 2.0
    _view_matrix: np.ndarray = np.eye(4)
    _projection_matrix: np.ndarray = np.eye(4)
    _camera_pos: np.ndarray = np.array([0.0, 0.0, 3.0], dtype=np.float32)
    _camera_front: np.ndarray = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    _camera_speed: float = 0.05
    _frame_speed = 1.0
    _render_new_frame: bool = True

    def __init__(self, show_origin: bool = True):
        self.create_framebuffers()
        # self._setup_framebuffer()
        self.reset_camera()
        self.show_origin = show_origin
        self.origin_renderer = AxisRenderer()
        self.origin_renderer.update(
            positions=torch.tensor([[0.0, 0.0, 0.0]]).cuda(),
            quats=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).cuda(),
        )
        self.axis_renderer = AxisRenderer()


    def create_framebuffers(self):
        self._frame_texture = None
        self._frame_depth_texture = None
        self._frame_fbo = None
        self._frame_pbo = None

        self._frame_vertices = np.array(
            [
                # Positions  TexCoords
                -1.0,
                -1.0,
                0.0,
                0.0,
                1.0,
                -1.0,
                1.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                -1.0,
                1.0,
                0.0,
                1.0,
            ],
            dtype=np.float32,
        )
        self._frame_indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        self._frame_vao = gl.GLuint()
        gl.glGenVertexArrays(1, self._frame_vao)
        gl.glBindVertexArray(self._frame_vao)
        self._frame_vbo = gl.GLuint()
        gl.glGenBuffers(1, self._frame_vbo)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._frame_vbo)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER,
            self._frame_vertices.nbytes,
            self._frame_vertices.ctypes.data,
            gl.GL_STATIC_DRAW,
        )

        self._frame_ebo = gl.GLuint()
        gl.glGenBuffers(1, self._frame_ebo)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._frame_ebo)
        gl.glBufferData(
            gl.GL_ELEMENT_ARRAY_BUFFER,
            self._frame_indices.nbytes,
            self._frame_indices.ctypes.data,
            gl.GL_STATIC_DRAW,
        )

        gl.glVertexAttribPointer(
            0,
            2,
            gl.GL_FLOAT,
            gl.GL_FALSE,
            4 * self._frame_vertices.itemsize,
            ctypes.c_void_p(0),
        )
        gl.glEnableVertexAttribArray(0)
        float_size = ctypes.sizeof(ctypes.c_float)
        gl.glVertexAttribPointer(
            1,
            2,
            gl.GL_FLOAT,
            gl.GL_FALSE,
            4 * self._frame_vertices.itemsize,
            ctypes.c_void_p(2 * float_size),
        )
        gl.glEnableVertexAttribArray(1)
        self._frame_shader = ShaderProgram(
            Shader(frame_vertex_shader, "vertex"),
            Shader(frame_fragment_shader, "fragment"),
        )
        gl.glUseProgram(self._frame_shader.id)
        self._frame_loc_texture = gl.glGetUniformLocation(
            self._frame_shader.id, str_buffer("textureSampler")
        )

        self._frame_depth_shader = ShaderProgram(
            Shader(frame_vertex_shader, "vertex"),
            Shader(frame_depth_fragment_shader, "fragment"),
        )
        gl.glUseProgram(self._frame_depth_shader.id)
        self._frame_loc_depth_texture = gl.glGetUniformLocation(
            self._frame_depth_shader.id, str_buffer("textureSampler")
        )

        # unbind the VBO and VAO
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

    def _setup_framebuffer(self):
        if self._frame_texture is None:
            self._frame_texture = gl.GLuint()
            gl.glGenTextures(1, self._frame_texture)
        if self._frame_depth_texture is None:
            self._frame_depth_texture = gl.GLuint()
            gl.glGenTextures(1, self._frame_depth_texture)

        # set up RGB texture
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._frame_texture)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGB,
            self.screen_width,
            self.screen_height,
            0,
            gl.GL_RGB,
            gl.GL_UNSIGNED_BYTE,
            None,
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)

        # set up depth texture
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._frame_depth_texture)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_DEPTH_COMPONENT32,
            self.screen_width,
            self.screen_height,
            0,
            gl.GL_DEPTH_COMPONENT,
            gl.GL_FLOAT,
            None,
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)


        # create a framebuffer object (FBO)
        if self._frame_fbo is None:
            self._frame_fbo = gl.GLuint()
            gl.glGenFramebuffers(1, self._frame_fbo)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._frame_fbo)

            # attach the texture to the FBO as its color attachment
            gl.glFramebufferTexture2D(
                gl.GL_FRAMEBUFFER,
                gl.GL_COLOR_ATTACHMENT0,
                gl.GL_TEXTURE_2D,
                self._frame_texture,
                0,
            )
            # attach the depth texture to the FBO as its depth attachment
            gl.glFramebufferTexture2D(
                gl.GL_FRAMEBUFFER,
                gl.GL_DEPTH_ATTACHMENT,
                gl.GL_TEXTURE_2D,
                self._frame_depth_texture,
                0,
            )

            if (
                gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
                != gl.GL_FRAMEBUFFER_COMPLETE
            ):
                print("Framebuffer is not complete!")
                gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
                sys.exit(1)

        # unbind the FBO (switch back to the default framebuffer)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        if self._frame_pbo is None:
            self._frame_pbo = gl.GLuint()
            gl.glGenBuffers(1, self._frame_pbo)  # generate 1 buffer reference
        gl.glBindBuffer(
            gl.GL_PIXEL_PACK_BUFFER, self._frame_pbo
        )  # binding to this buffer

        # allocate memory for PBO
        rgb_bytes_per_pixel = 3
        depth_bytes_per_pixel = 4
        pixels = np.zeros(
            (
                self.screen_height,
                self.screen_width,
                rgb_bytes_per_pixel + depth_bytes_per_pixel,
            ),
            dtype=np.uint8,
        )
        gl.glBufferData(
            gl.GL_PIXEL_PACK_BUFFER,
            pixels.nbytes,
            pixels.ctypes.data,
            gl.GL_DYNAMIC_DRAW,
        )
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)

    def go_to_view(
        self,
        x_wv: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        w: int,
        h: int,
    ):
        self.fl_x = fx
        self.fl_y = fy
        self.screen_center_x = cx / w
        self.screen_center_y = cy / h

        cam_pos = PyVec3(*(x_wv[:3, 3]))
        self._camera_pos = cam_pos
        self._camera_front = PyVec3(*(-x_wv[:3, 2]))
        self._camera_up = PyVec3(*(x_wv[:3, 1]))
        x_vw = np.linalg.inv(x_wv)
        self.update_projection_matrix()
        self._view_matrix = x_vw.T.flatten()

    def update_view_matrix(self):
        cam_pos = self._camera_pos
        self._view_matrix = np.array(
            PyMat4.look_at(cam_pos, cam_pos + self._camera_front, self._camera_up),
            dtype=np.float32,
        )
        self._render_new_frame = True

    def update_projection_matrix(self):
        if self.screen_height == 0:
            return
        self._projection_matrixT = self.gl_projectionT()
        K = self.K()
        vals = calibration_matrix_values(K, (self.screen_width, self.screen_height))
        fov_x = vals['fovx']
        fov_y = vals['fovy']
        self.fov_x = fov_x * np.pi / 180.0
        self.fov_y = fov_y * np.pi / 180.0

    def gl_projectionT(self) -> np.ndarray:
        return convert_K_to_projection_matrixT(
            self.K(), self.screen_width, self.screen_height
        )    

    def gl_projection_from_world(self) -> np.ndarray:
        x_vw = self.x_vw()
        return self._projection_matrixT.T @ x_vw

    def K(self) -> np.ndarray:
        cx = self.screen_width * self.screen_center_x
        cy = self.screen_height * self.screen_center_y
        fl_x = self.fl_x
        fl_y = self.fl_y
        return np.array(
            [
                [fl_x, 0, cx],
                [0, fl_y, cy],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

    @property
    def aspect(self) -> float:
        return self.screen_width / self.screen_height

    def x_vw(self) -> np.ndarray:
        x_vw = self._view_matrix.reshape((4, 4)).T
        return x_vw

    def x_wv(self) -> np.ndarray:
        x_vw = self.x_vw()
        x_wv = np.linalg.inv(x_vw)
        return x_wv

    def reset_camera(self):
        self._camera_pos = PyVec3(0.0, -2.0, 0.4)
        self._camera_front = PyVec3(0.0, 1.0, 0.0)
        self._camera_up = PyVec3(0.0, 0.0, 1.0)
        self._render_new_frame = True
        self.update_view_matrix()
        self.update_projection_matrix()

    def process_nav(self):
        self.process_mouse()
        if (
            imgui.is_key_down(imgui.Key.w)
            or imgui.is_key_down(imgui.Key.up_arrow)
        ):
            self._camera_pos += self._camera_front * (self._camera_speed)
            self.update_view_matrix()
        if (
            imgui.is_key_down(imgui.Key.s)
            or imgui.is_key_down(imgui.Key.down_arrow)
        ):
            self._camera_pos -= self._camera_front * (self._camera_speed)
            self.update_view_matrix()
        if (
            imgui.is_key_down(imgui.Key.a)
            or imgui.is_key_down(imgui.Key.left_arrow)
        ):
            camera_side = PyVec3.cross(self._camera_front, self._camera_up).normalize()
            self._camera_pos -= camera_side * (self._camera_speed)
            self.update_view_matrix()
        if (
            imgui.is_key_down(imgui.Key.d)
            or imgui.is_key_down(imgui.Key.right_arrow)
        ):
            camera_side = PyVec3.cross(self._camera_front, self._camera_up).normalize()
            self._camera_pos += camera_side * (self._camera_speed)
            self.update_view_matrix()

    def process_mouse(self):
        dx=imgui.get_io().mouse_delta.x
        dy=-imgui.get_io().mouse_delta.y
        scroll=-imgui.get_io().mouse_wheel*2
        buttons=imgui.get_io().mouse_down
        shift=imgui.get_io().key_shift
        ctrl=imgui.get_io().key_ctrl
        if shift:
            sensitivity = 0.01
        else:
            sensitivity = 0.1
        if buttons[2]:
            dx *= self._camera_speed * self._frame_speed * sensitivity
            dy *= self._camera_speed * self._frame_speed * sensitivity

            camera_side = PyVec3.cross(self._camera_front, self._camera_up).normalize()
            camera_up = PyVec3.cross(camera_side, self._camera_front).normalize()

            if ctrl:
                self._camera_pos -= self._camera_front * dy
            else:
                self._camera_pos -= camera_up * dy
                self._camera_pos -= camera_side * dx
            self.update_view_matrix()
        # get draw duration

        if buttons[0]:
            dx *= sensitivity
            dy *= sensitivity

            # orbit camera
            from scipy.spatial.transform import Rotation as R

            r = R.from_euler("xyz", [dy, -dx, 0], degrees=True)
            r_vw = np.array(self._view_matrix).reshape((4, 4)).T[:3, :3]
            r_wv = np.linalg.inv(r_vw)
            r_wv = r_wv @ r.as_matrix()
            front = r_wv @ np.array([0, 0, -1])

            self._camera_front = PyVec3(*front)
            self.update_view_matrix()

        if scroll:
            sensitivity = 10
            if shift:
                sensitivity = 100
            self.fl_x += scroll * sensitivity
            self.fl_y += scroll * sensitivity
            self.fl_x = max(1.0, self.fl_x)
            self.fl_y = max(1.0, self.fl_y)

    def world2projT(self):
        viewT = np.asarray(self._view_matrix, dtype=np.float32).reshape((4, 4))
        projT = np.asarray(self._projection_matrixT, dtype=np.float32).reshape((4, 4))
        return viewT @ projT  # These are all column major

    def axes(self, name: str, positions: torch.Tensor, quats: torch.Tensor):
        renderer = self.axis_renderer
        renderer.update(positions, quats)
        renderer.draw(self.world2projT().flatten())

    def imgui_active(self):
        return (
            imgui.is_any_item_hovered()
            or imgui.is_any_item_focused()
            or imgui.is_any_item_active()
            or imgui.get_io().want_capture_mouse
        )

    def begin(self, name: str = ""):
        self.name = name
        if name:
            imgui.begin(name)
            ds = imgui.get_content_region_avail()
        else:
            ds = imgui.get_io().display_size

        screen_width, screen_height = int(ds.x), int(ds.y)

        if screen_height == 0 or screen_width == 0:
            self.screen_width, self.screen_height = screen_width, screen_height
            return np.eye(4, dtype=np.float32).flatten()

        if screen_width != self.screen_width or screen_height != self.screen_height:
            self.screen_width, self.screen_height = screen_width, screen_height
            self._setup_framebuffer()

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._frame_fbo)
        gl.glClearColor(*self.background_color, 1.0)
        self.update_projection_matrix()
        world2projT = self.world2projT().flatten()
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LESS)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        if self.show_origin:
            self.origin_renderer.draw(
                world2projT=world2projT,
                line_width=self.line_width,
                scale=0.1,
            )
        return world2projT
    
    def end(self):
        if self.screen_width == 0 or self.screen_height == 0:
            if self.name:
                imgui.end()
            return
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glViewport(0, 0, self.screen_width, self.screen_height)
        if self.name:
            imgui.image(
                    self._frame_texture.value, 
                    ImVec2(self.screen_width, self.screen_height),
                    uv0=ImVec2(0, 1),
                    uv1=ImVec2(1, 0))
            if self.imgui_active() and imgui.is_item_hovered():
                self.process_nav()
            imgui.end()
        else:
            with self._frame_shader:
                gl.glActiveTexture(gl.GL_TEXTURE0)
                gl.glBindTexture(gl.GL_TEXTURE_2D, self._frame_texture)
                gl.glUniform1i(self._frame_loc_texture, 0)
                gl.glBindVertexArray(self._frame_vao)
                gl.glDrawElements(
                    gl.GL_TRIANGLES, len(self._frame_indices), gl.GL_UNSIGNED_INT, None
                )
                gl.glBindVertexArray(0)
                gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            if not self.imgui_active():
                self.process_nav()
