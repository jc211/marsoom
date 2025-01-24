import ctypes
import math
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Tuple, Literal

import numpy as np
import pyglet
import torch
from imgui_bundle import ImVec2, imgui, imguizmo
from pyglet import gl
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.math import Mat4 as PyMat4
from pyglet.math import Vec3 as PyVec3

from marsoom.axes import Axes
from marsoom.context_3d import Context3D
from marsoom.utils import (
    calibration_matrix_values,
    convert_K_to_projection_matrixT,
    ortho_matrix_T,
    str_buffer,
)

guizmo = imguizmo.im_guizmo


def get_default_shader() -> ShaderProgram:
    return ShaderProgram(Shader(default_vertex_source, "vertex"), Shader(default_fragment_source, "fragment"))

default_vertex_source = """#version 150 core
    in vec4 position;

    uniform WindowBlock
    {
        mat4 projection;
        mat4 view;
    } window;

    uniform ViewportBlock {
        float width;
        float height;
    } viewport;

    uniform LightBlock {
        vec3 viewPos;
        vec3 lightColor;
        vec3 sunDirection;
    } light;


    void main()
    {
        gl_Position = window.projection * window.view * position;
    }
"""
default_fragment_source = """#version 150 core
    out vec4 color;

    void main()
    {
        color = vec4(1.0, 0.0, 0.0, 1.0);
    }
"""


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


class Viewer3D:
    screen_width: int = 0
    screen_height: int = 0
    fl_x: float = 500.0
    fl_y: float = 500.0
    ortho_zoom: float = 1.0
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

    zoom_sensitivity: float = 0.1

    orthogonal: bool = False

    def __init__(self, window, show_origin: bool = True):
        self.create_framebuffers()
        # self._setup_framebuffer()

        self._default_shader = get_default_shader()
        self._light_block = self._default_shader.uniform_blocks["LightBlock"].create_ubo()
        self._viewport_block = self._default_shader.uniform_blocks["ViewportBlock"].create_ubo()
        with self._light_block as ubo:
            ubo.lightColor[:] = (1.0, 1.0, 1.0)
            ubo.sunDirection[:] = (0.0, 0.0, 1.0)
        

        self.reset_camera()
        self.window = window
        self.show_origin = show_origin
        self.in_imgui_window = False
        self.window_draw_list = None
        self.tl = None
        self._batch = pyglet.graphics.Batch()
        self._origin = Axes(batch=self._batch)
        self.near = 0.01
        self.far = 100.0

    def create_framebuffers(self):
        self._frame_texture = None
        self._frame_depth_texture = None
        self._frame_fbo = None
        self._frame_pbo = None

        self._frame_vertices = np.array(
            [
                # Positions  TexCoords
                -1.0, -1.0, 0.0, 0.0, 
                1.0, -1.0, 1.0, 0.0, 
                1.0, 1.0, 1.0, 1.0, 
                -1.0, 1.0, 0.0, 1.0,
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
        
        with self._viewport_block as ubo:
            ubo.width[0] = ctypes.c_float(float(self.screen_width))
            ubo.height[0] = ctypes.c_float(float(self.screen_height))

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
    
    def set_sun_direction(self, sun_direction: np.ndarray):
        with self._light_block as ubo:
            ubo.sunDirection[:] = sun_direction
    
    def set_light_color(self, light_color: np.ndarray):
        with self._light_block as ubo:
            ubo.lightColor[:] = light_color

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
        with self._light_block as ubo:
            ubo.viewPos[:] = cam_pos
        self._render_new_frame = True

    def update_projection_matrix(self):
        if self.screen_height == 0:
            return
        self._projection_matrixT = self.gl_projectionT()
        K = self.K()
        vals = calibration_matrix_values(K, (self.screen_width, self.screen_height))
        fov_x = vals["fovx"]
        fov_y = vals["fovy"]
        self.fov_x = fov_x * np.pi / 180.0
        self.fov_y = fov_y * np.pi / 180.0

    def gl_projectionT(self) -> np.ndarray:
        if self.orthogonal:
            return ortho_matrix_T(
                self.ortho_zoom, self.screen_width, self.screen_height, self.near, self.far
            )
        else:
            return convert_K_to_projection_matrixT(
                self.K(), self.screen_width, self.screen_height, self.near, self.far
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

    def x_vw(self, standard: Literal['opencv', 'blender'] = 'blender') -> np.ndarray:
        x_vw = self._view_matrix.reshape((4, 4)).T
        if standard == 'opencv':
            x_vw = np.array(
                [
                    [1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1],
                ],
                dtype=np.float32,
            ) @ x_vw
        return x_vw

    def x_wv(self, standard: Literal['opencv', 'blender'] = 'blender') -> np.ndarray:
        x_vw = self.x_vw(standard=standard)
        x_wv = np.linalg.inv(x_vw)
        return x_wv

    def manipulate(
        self,
        object_matrix: np.array,
        operation: guizmo.OPERATION = guizmo.OPERATION.translate,
        mode: guizmo.MODE = guizmo.MODE.local,
    ):

        if self.in_imgui_window:
            if self.tl is not None:
                guizmo.set_drawlist(self.window_draw_list)
                guizmo.set_rect(
                    self.tl.x, self.tl.y, self.screen_width, self.screen_height
                )
        else:
            guizmo.set_rect(0, 0, self.screen_width, self.screen_height)

        guizmo.allow_axis_flip(False)
        proj_matrix = guizmo.Matrix16(self.gl_projectionT().flatten())
        view_matrix = guizmo.Matrix16(self._view_matrix.flatten())

        object_matrix = guizmo.Matrix16(object_matrix.T.flatten())

        changed = guizmo.manipulate(
            view=view_matrix,
            projection=proj_matrix,
            operation=operation,
            mode=mode,
            object_matrix=object_matrix,
        )
        res_matrix = np.array(object_matrix.values).reshape((4, 4)).T
        return changed, res_matrix

    def reset_camera(self):
        self._camera_pos = PyVec3(0.0, -2.0, 0.4)
        self._camera_front = PyVec3(0.0, 1.0, 0.0)
        self._camera_up = PyVec3(0.0, 0.0, 1.0)
        self._render_new_frame = True
        self.update_view_matrix()
        self.update_projection_matrix()

    def top_view(self):
        self.orthogonal = True
        self._camera_pos = PyVec3(0.0, 0.0, 2.0)
        self._camera_front = PyVec3(0.0, 0.0, -1.0)
        self._camera_up = PyVec3(0.0, 1.0, 0.0)
        self.update_view_matrix()
        self.update_projection_matrix()

    def bottom_view(self):
        self.orthogonal = True
        self._camera_pos = PyVec3(0.0, 0.0, -2.0)
        self._camera_front = PyVec3(0.0, 0.0, 1.0)
        self._camera_up = PyVec3(0.0, 1.0, 0.0)
        self.update_view_matrix()
        self.update_projection_matrix()

    def front_view(self):
        self.orthogonal = True
        self._camera_pos = PyVec3(0.0, 2.0, 0.0)
        self._camera_front = PyVec3(0.0, -1.0, 0.0)
        self._camera_up = PyVec3(0.0, 0.0, 1.0)
        self.update_view_matrix()
        self.update_projection_matrix()

    def back_view(self):
        self.orthogonal = True
        self._camera_pos = PyVec3(0.0, -2.0, 0.0)
        self._camera_front = PyVec3(0.0, 1.0, 0.0)
        self._camera_up = PyVec3(0.0, 0.0, 1.0)
        self.update_view_matrix()
        self.update_projection_matrix()

    def right_view(self):
        self.orthogonal = True
        self._camera_pos = PyVec3(2.0, 0.0, 0.0)
        self._camera_front = PyVec3(-1.0, 0.0, 0.0)
        self._camera_up = PyVec3(0.0, 0.0, 1.0)
        self.update_view_matrix()
        self.update_projection_matrix()

    def left_view(self):
        self.orthogonal = True
        self._camera_pos = PyVec3(-2.0, 0.0, 0.0)
        self._camera_front = PyVec3(1.0, 0.0, 0.0)
        self._camera_up = PyVec3(0.0, 0.0, 1.0)
        self.update_view_matrix()
        self.update_projection_matrix()

    def reset_view(self):
        self.orthogonal = False
        self.reset_camera()

    def process_nav(self):
        if self.in_imgui_window:
            if not (self.imgui_active() and imgui.is_item_hovered()):
                return
        else:
            if self.imgui_active():
                return
        self.process_mouse()
        ctrl = imgui.get_io().key_ctrl
        if ctrl:
            return
        if imgui.is_key_down(imgui.Key.w) or imgui.is_key_down(imgui.Key.up_arrow):
            self.orthogonal = False
            self._camera_pos += self._camera_front * (self._camera_speed)
            self.update_view_matrix()
        if imgui.is_key_down(imgui.Key.s) or imgui.is_key_down(imgui.Key.down_arrow):
            self.orthogonal = False
            self._camera_pos -= self._camera_front * (self._camera_speed)
            self.update_view_matrix()
        if imgui.is_key_down(imgui.Key.a) or imgui.is_key_down(imgui.Key.left_arrow):
            camera_side = PyVec3.cross(self._camera_front, self._camera_up).normalize()
            self._camera_pos -= camera_side * (self._camera_speed)
            self.update_view_matrix()
        if imgui.is_key_down(imgui.Key.d) or imgui.is_key_down(imgui.Key.right_arrow):
            camera_side = PyVec3.cross(self._camera_front, self._camera_up).normalize()
            self._camera_pos += camera_side * (self._camera_speed)
            self.update_view_matrix()

    def process_mouse(self):
        if guizmo.is_using_any():
            return
        io = imgui.get_io()
        dx = io.mouse_delta.x
        dy = -io.mouse_delta.y
        scroll = -io.mouse_wheel * 2
        buttons = io.mouse_down
        shift = io.key_shift
        ctrl = io.key_ctrl
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

            self.orthogonal = False

            r = R.from_euler("xyz", [dy, -dx, 0], degrees=True)
            r_vw = np.array(self._view_matrix).reshape((4, 4)).T[:3, :3]
            r_wv = np.linalg.inv(r_vw)
            r_wv = r_wv @ r.as_matrix()
            front = r_wv @ np.array([0, 0, -1])

            self._camera_front = PyVec3(*front)
            self.update_view_matrix()

        if scroll:
            sensitivity = self.zoom_sensitivity
            if shift:
                sensitivity = 10 * sensitivity
            if self.orthogonal:
                sensitivity = 1
                self.ortho_zoom *= 1.0 + scroll * sensitivity
                self.ortho_zoom = max(1.0, self.ortho_zoom)
                self.update_projection_matrix()
            else:
                self.fl_x += scroll * sensitivity
                self.fl_y += scroll * sensitivity
                self.fl_x = max(1.0, self.fl_x)
                self.fl_y = max(1.0, self.fl_y)

    def world2projT(self):
        viewT = np.asarray(self._view_matrix, dtype=np.float32).reshape((4, 4))
        projT = np.asarray(self._projection_matrixT, dtype=np.float32).reshape((4, 4))
        return viewT @ projT  # These are all column major

    def imgui_active(self):
        return (
            imgui.is_any_item_hovered()
            or imgui.is_any_item_focused()
            or imgui.is_any_item_active()
            or imgui.get_io().want_capture_mouse
        )
    

    @contextmanager
    def draw(self, in_imgui_window: bool = False):
        # check if in imgui window
        self.in_imgui_window = in_imgui_window
        if self.in_imgui_window:
            ds = imgui.get_content_region_avail()
            self.in_imgui_window = True
            self.window_draw_list = imgui.get_window_draw_list()
        else:
            ds = imgui.get_io().display_size
            self.in_imgui_window = False

        screen_width, screen_height = int(ds.x), int(ds.y)
        screen_height = max(screen_height, 1)
        screen_width = max(screen_width, 1)

        if screen_width != self.screen_width or screen_height != self.screen_height:
            self.screen_width, self.screen_height = screen_width, screen_height
            self._setup_framebuffer()

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._frame_fbo)
        gl.glClearColor(*self.background_color, 1.0)
        self.update_projection_matrix()
        world2projT = self.world2projT().flatten()
        self.window.projection = self._projection_matrixT.flatten()
        self.window.view = self._view_matrix.T.flatten()

        gl.glEnable(gl.GL_DEPTH_TEST)
        # gl.glEnable(gl.GL_CULL_FACE)
        gl.glDepthFunc(gl.GL_LESS)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        context = Context3D(world2projT=world2projT, camera_positon=self._camera_pos)

        self._batch.draw()
        yield context

        if self.screen_width == 0 or self.screen_height == 0:
            return

        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glViewport(0, 0, self.screen_width, self.screen_height)
        if self.window_draw_list:
            self.tl = imgui.get_cursor_screen_pos()
            imgui.image(
                self._frame_texture.value,
                ImVec2(self.screen_width, self.screen_height),
                uv0=ImVec2(0, 1),
                uv1=ImVec2(1, 0),
            )
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
