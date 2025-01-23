# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import ctypes
from collections import defaultdict
from typing import List, Tuple, Union
import pyglet.gl as gl
from pyglet.graphics.shader import ShaderProgram

import numpy as np

import warp as wp

Mat44 = Union[List[float], List[List[float]], np.ndarray]

def tab10_color_map(i):
    # matplotlib "tab10" colors
    colors = [
        [31, 119, 180],
        [255, 127, 14],
        [44, 160, 44],
        [214, 39, 40],
        [148, 103, 189],
        [140, 86, 75],
        [227, 119, 194],
        [127, 127, 127],
        [188, 189, 34],
        [23, 190, 207],
    ]
    num_colors = len(colors)
    return [c / 255.0 for c in colors[i % num_colors]]



wp.set_module_options({"enable_backward": False})
def get_default_shader() -> ShaderProgram:
    return gl.current_context.create_program((shape_vertex_shader, 'vertex'),
                                             (shape_fragment_shader, 'fragment'))


shape_vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

// column vectors of the instance transform matrix
layout (location = 3) in vec4 aInstanceTransform0;
layout (location = 4) in vec4 aInstanceTransform1;
layout (location = 5) in vec4 aInstanceTransform2;
layout (location = 6) in vec4 aInstanceTransform3;

// colors to use for the checkerboard pattern
layout (location = 7) in vec3 aObjectColor1;
layout (location = 8) in vec3 aObjectColor2;

uniform WindowBlock
{
    mat4 projection;
    mat4 view;
} window;

out vec3 Normal;
out vec3 FragPos;
out vec2 TexCoord;
out vec3 ObjectColor1;
out vec3 ObjectColor2;

void main()
{
    mat4 transform = mat4(aInstanceTransform0, aInstanceTransform1, aInstanceTransform2, aInstanceTransform3);
    vec4 worldPos = transform * vec4(aPos, 1.0);
    gl_Position = window.projection * window.view * worldPos;

    FragPos = vec3(worldPos);
    Normal = mat3(transpose(inverse(transform))) * aNormal;
    TexCoord = aTexCoord;
    ObjectColor1 = aObjectColor1;
    ObjectColor2 = aObjectColor2;
}
"""

shape_fragment_shader = """
#version 330 core
out vec4 FragColor;

in vec3 Normal;
in vec3 FragPos;
in vec2 TexCoord;
in vec3 ObjectColor1;
in vec3 ObjectColor2;

uniform LightBlock {
    vec3 viewPos;
    vec3 lightColor;
    vec3 sunDirection;
} light;


void main()
{
    float ambientStrength = 0.3;
    vec3 ambient = ambientStrength * light.lightColor;
    vec3 norm = normalize(Normal);

    float diff = max(dot(norm, light.sunDirection), 0.0);
    vec3 diffuse = diff * light.lightColor;

    vec3 lightDir2 = normalize(vec3(1.0, 0.3, -0.3));
    diff = max(dot(norm, lightDir2), 0.0);
    diffuse += diff * light.lightColor * 0.3;

    float specularStrength = 0.5;
    vec3 viewDir = normalize(light.viewPos - FragPos);

    vec3 reflectDir = reflect(-light.sunDirection, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * light.lightColor;

    reflectDir = reflect(-lightDir2, norm);
    spec = pow(max(dot(viewDir, reflectDir), 0.0), 64);
    specular += specularStrength * spec * light.lightColor * 0.3;

    // checkerboard pattern
    float u = TexCoord.x;
    float v = TexCoord.y;
    // blend the checkerboard pattern dependent on the gradient of the texture coordinates
    // to void Moire patterns
    vec2 grad = abs(dFdx(TexCoord)) + abs(dFdy(TexCoord));
    float blendRange = 1.5;
    float blendFactor = max(grad.x, grad.y) * blendRange;
    float scale = 2.0;
    float checker = mod(floor(u * scale) + floor(v * scale), 2.0);
    checker = mix(checker, 0.5, smoothstep(0.0, 1.0, blendFactor));
    vec3 checkerColor = mix(ObjectColor1, ObjectColor2, checker);

    vec3 result = (ambient + diffuse + specular) * checkerColor;
    FragColor = vec4(result, 1.0);
}
"""



@wp.kernel
def update_vbo_transforms(
    instance_id: wp.array(dtype=int),
    instance_body: wp.array(dtype=int),
    instance_transforms: wp.array(dtype=wp.transform),
    instance_scalings: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    # outputs
    vbo_transforms: wp.array(dtype=wp.mat44),
):
    tid = wp.tid()
    i = instance_id[tid]
    X_ws = instance_transforms[i]
    if instance_body:
        body = instance_body[i]
        if body >= 0:
            if body_q:
                X_ws = body_q[body] * X_ws
            else:
                return
    p = wp.transform_get_translation(X_ws)
    q = wp.transform_get_rotation(X_ws)
    s = instance_scalings[i]
    rot = wp.quat_to_matrix(q)
    # transposed definition
    vbo_transforms[tid] = wp.mat44(
        rot[0, 0] * s[0],
        rot[1, 0] * s[0],
        rot[2, 0] * s[0],
        0.0,
        rot[0, 1] * s[1],
        rot[1, 1] * s[1],
        rot[2, 1] * s[1],
        0.0,
        rot[0, 2] * s[2],
        rot[1, 2] * s[2],
        rot[2, 2] * s[2],
        0.0,
        p[0],
        p[1],
        p[2],
        1.0,
    )


@wp.kernel
def update_vbo_vertices(
    points: wp.array(dtype=wp.vec3),
    scale: wp.vec3,
    # outputs
    vbo_vertices: wp.array(dtype=float, ndim=2),
):
    tid = wp.tid()
    p = points[tid]
    vbo_vertices[tid, 0] = p[0] * scale[0]
    vbo_vertices[tid, 1] = p[1] * scale[1]
    vbo_vertices[tid, 2] = p[2] * scale[2]


@wp.kernel
def update_points_positions(
    instance_positions: wp.array(dtype=wp.vec3),
    instance_scalings: wp.array(dtype=wp.vec3),
    # outputs
    vbo_transforms: wp.array(dtype=wp.mat44),
):
    tid = wp.tid()
    p = instance_positions[tid]
    s = wp.vec3(1.0)
    if instance_scalings:
        s = instance_scalings[tid]
    # transposed definition
    # fmt: off
    vbo_transforms[tid] = wp.mat44(
        s[0],  0.0,  0.0, 0.0,
         0.0, s[1],  0.0, 0.0,
         0.0,  0.0, s[2], 0.0,
        p[0], p[1], p[2], 1.0)
    # fmt: on


@wp.kernel
def update_line_transforms(
    lines: wp.array(dtype=wp.vec3, ndim=2),
    # outputs
    vbo_transforms: wp.array(dtype=wp.mat44),
):
    tid = wp.tid()
    p0 = lines[tid, 0]
    p1 = lines[tid, 1]
    p = 0.5 * (p0 + p1)
    d = p1 - p0
    s = wp.length(d)
    axis = wp.normalize(d)
    y_up = wp.vec3(0.0, 1.0, 0.0)
    angle = wp.acos(wp.dot(axis, y_up))
    axis = wp.normalize(wp.cross(axis, y_up))
    q = wp.quat_from_axis_angle(axis, -angle)
    rot = wp.quat_to_matrix(q)
    # transposed definition
    # fmt: off
    vbo_transforms[tid] = wp.mat44(
            rot[0, 0],     rot[1, 0],     rot[2, 0], 0.0,
        s * rot[0, 1], s * rot[1, 1], s * rot[2, 1], 0.0,
            rot[0, 2],     rot[1, 2],     rot[2, 2], 0.0,
                 p[0],          p[1],          p[2], 1.0,
    )
    # fmt: on


@wp.kernel
def compute_gfx_vertices(
    indices: wp.array(dtype=int, ndim=2),
    vertices: wp.array(dtype=wp.vec3, ndim=1),
    scale: wp.vec3,
    # outputs
    gfx_vertices: wp.array(dtype=float, ndim=2),
):
    tid = wp.tid()
    v0 = vertices[indices[tid, 0]] * scale[0]
    v1 = vertices[indices[tid, 1]] * scale[1]
    v2 = vertices[indices[tid, 2]] * scale[2]
    i = tid * 3
    j = i + 1
    k = i + 2
    gfx_vertices[i, 0] = v0[0]
    gfx_vertices[i, 1] = v0[1]
    gfx_vertices[i, 2] = v0[2]
    gfx_vertices[j, 0] = v1[0]
    gfx_vertices[j, 1] = v1[1]
    gfx_vertices[j, 2] = v1[2]
    gfx_vertices[k, 0] = v2[0]
    gfx_vertices[k, 1] = v2[1]
    gfx_vertices[k, 2] = v2[2]
    n = wp.normalize(wp.cross(v1 - v0, v2 - v0))
    gfx_vertices[i, 3] = n[0]
    gfx_vertices[i, 4] = n[1]
    gfx_vertices[i, 5] = n[2]
    gfx_vertices[j, 3] = n[0]
    gfx_vertices[j, 4] = n[1]
    gfx_vertices[j, 5] = n[2]
    gfx_vertices[k, 3] = n[0]
    gfx_vertices[k, 4] = n[1]
    gfx_vertices[k, 5] = n[2]


@wp.kernel
def compute_average_normals(
    indices: wp.array(dtype=int, ndim=2),
    vertices: wp.array(dtype=wp.vec3),
    scale: wp.vec3,
    # outputs
    normals: wp.array(dtype=wp.vec3),
    faces_per_vertex: wp.array(dtype=int),
):
    tid = wp.tid()
    i = indices[tid, 0]
    j = indices[tid, 1]
    k = indices[tid, 2]
    v0 = vertices[i] * scale[0]
    v1 = vertices[j] * scale[1]
    v2 = vertices[k] * scale[2]
    n = wp.normalize(wp.cross(v1 - v0, v2 - v0))
    wp.atomic_add(normals, i, n)
    wp.atomic_add(faces_per_vertex, i, 1)
    wp.atomic_add(normals, j, n)
    wp.atomic_add(faces_per_vertex, j, 1)
    wp.atomic_add(normals, k, n)
    wp.atomic_add(faces_per_vertex, k, 1)


@wp.kernel
def assemble_gfx_vertices(
    vertices: wp.array(dtype=wp.vec3, ndim=1),
    normals: wp.array(dtype=wp.vec3),
    faces_per_vertex: wp.array(dtype=int),
    scale: wp.vec3,
    # outputs
    gfx_vertices: wp.array(dtype=float, ndim=2),
):
    tid = wp.tid()
    v = vertices[tid]
    n = normals[tid] / float(faces_per_vertex[tid])
    gfx_vertices[tid, 0] = v[0] * scale[0]
    gfx_vertices[tid, 1] = v[1] * scale[1]
    gfx_vertices[tid, 2] = v[2] * scale[2]
    gfx_vertices[tid, 3] = n[0]
    gfx_vertices[tid, 4] = n[1]
    gfx_vertices[tid, 5] = n[2]



def check_gl_error():
    from pyglet import gl

    error = gl.glGetError()
    if error != gl.GL_NO_ERROR:
        print(f"OpenGL error: {error}")


class ShapeInstancer:
    """
    Handles instanced rendering for a mesh.
    Note the vertices must be in the 8-dimensional format:
        [3D point, 3D normal, UV texture coordinates]
    """

    def __new__(cls, *args, **kwargs):
        instance = super(ShapeInstancer, cls).__new__(cls)
        instance.instance_transform_gl_buffer = None
        instance.vao = None
        return instance

    def __init__(self, shape_shader, device):
        self.shape_shader = shape_shader
        self.device = device
        self.face_count = 0
        self.instance_color1_buffer = None
        self.instance_color2_buffer = None
        self.color1 = (1.0, 1.0, 1.0)
        self.color2 = (0.0, 0.0, 0.0)
        self.num_instances = 0
        self.transforms = None
        self.scalings = None
        self._instance_transform_cuda_buffer = None

    def __del__(self):
        from pyglet import gl

        if self.instance_transform_gl_buffer is not None:
            try:
                gl.glDeleteBuffers(1, self.instance_transform_gl_buffer)
                gl.glDeleteBuffers(1, self.instance_color1_buffer)
                gl.glDeleteBuffers(1, self.instance_color2_buffer)
            except gl.GLException:
                pass
        if self.vao is not None:
            try:
                gl.glDeleteVertexArrays(1, self.vao)
                gl.glDeleteBuffers(1, self.vbo)
                gl.glDeleteBuffers(1, self.ebo)
            except gl.GLException:
                pass

    def register_shape(self, vertices, indices, color1=(1.0, 1.0, 1.0), color2=(0.0, 0.0, 0.0)):
        from pyglet import gl

        if color1 is not None and color2 is None:
            color2 = np.clip(np.array(color1) + 0.25, 0.0, 1.0)
        self.color1 = color1
        self.color2 = color2

        gl.glUseProgram(self.shape_shader.id)

        # Create VAO, VBO, and EBO
        self.vao = gl.GLuint()
        gl.glGenVertexArrays(1, self.vao)
        gl.glBindVertexArray(self.vao)

        self.vbo = gl.GLuint()
        gl.glGenBuffers(1, self.vbo)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices.ctypes.data, gl.GL_STATIC_DRAW)

        self.ebo = gl.GLuint()
        gl.glGenBuffers(1, self.ebo)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices.ctypes.data, gl.GL_STATIC_DRAW)

        # Set up vertex attributes
        vertex_stride = vertices.shape[1] * vertices.itemsize
        # positions
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, vertex_stride, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        # normals
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, vertex_stride, ctypes.c_void_p(3 * vertices.itemsize))
        gl.glEnableVertexAttribArray(1)
        # uv coordinates
        gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, vertex_stride, ctypes.c_void_p(6 * vertices.itemsize))
        gl.glEnableVertexAttribArray(2)

        gl.glBindVertexArray(0)

        self.face_count = len(indices)

    def update_colors(self, colors1, colors2):
        from pyglet import gl

        if colors1 is None:
            colors1 = np.tile(self.color1, (self.num_instances, 1))
        if colors2 is None:
            colors2 = np.tile(self.color2, (self.num_instances, 1))
        if np.shape(colors1) != (self.num_instances, 3):
            colors1 = np.tile(colors1, (self.num_instances, 1))
        if np.shape(colors2) != (self.num_instances, 3):
            colors2 = np.tile(colors2, (self.num_instances, 1))
        colors1 = np.array(colors1, dtype=np.float32)
        colors2 = np.array(colors2, dtype=np.float32)

        gl.glBindVertexArray(self.vao)

        # create buffer for checkerboard colors
        if self.instance_color1_buffer is None:
            self.instance_color1_buffer = gl.GLuint()
            gl.glGenBuffers(1, self.instance_color1_buffer)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_color1_buffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, colors1.nbytes, colors1.ctypes.data, gl.GL_STATIC_DRAW)

        if self.instance_color2_buffer is None:
            self.instance_color2_buffer = gl.GLuint()
            gl.glGenBuffers(1, self.instance_color2_buffer)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_color2_buffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, colors2.nbytes, colors2.ctypes.data, gl.GL_STATIC_DRAW)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_color1_buffer)
        gl.glVertexAttribPointer(7, 3, gl.GL_FLOAT, gl.GL_FALSE, colors1[0].nbytes, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(7)
        gl.glVertexAttribDivisor(7, 1)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_color2_buffer)
        gl.glVertexAttribPointer(8, 3, gl.GL_FLOAT, gl.GL_FALSE, colors2[0].nbytes, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(8)
        gl.glVertexAttribDivisor(8, 1)

    def allocate_instances(self, positions, rotations=None, colors1=None, colors2=None, scalings=None):
        from pyglet import gl

        gl.glBindVertexArray(self.vao)

        self.num_instances = len(positions)

        # Create instance buffer and bind it as an instanced array
        if self.instance_transform_gl_buffer is None:
            self.instance_transform_gl_buffer = gl.GLuint()
            gl.glGenBuffers(1, self.instance_transform_gl_buffer)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_transform_gl_buffer)

        self.instance_ids = wp.array(np.arange(self.num_instances), dtype=wp.int32, device=self.device)
        if rotations is None:
            self.instance_transforms = wp.array(
                [(*pos, 0.0, 0.0, 0.0, 1.0) for pos in positions], dtype=wp.transform, device=self.device
            )
        else:
            self.instance_transforms = wp.array(
                [(*pos, *rot) for pos, rot in zip(positions, rotations)],
                dtype=wp.transform,
                device=self.device,
            )

        if scalings is None:
            self.instance_scalings = wp.array(
                np.tile((1.0, 1.0, 1.0), (self.num_instances, 1)), dtype=wp.vec3, device=self.device
            )
        else:
            self.instance_scalings = wp.array(scalings, dtype=wp.vec3, device=self.device)

        vbo_transforms = wp.zeros(dtype=wp.mat44, shape=(self.num_instances,), device=self.device)

        wp.launch(
            update_vbo_transforms,
            dim=self.num_instances,
            inputs=[
                self.instance_ids,
                None,
                self.instance_transforms,
                self.instance_scalings,
                None,
            ],
            outputs=[
                vbo_transforms,
            ],
            device=self.device,
        )

        vbo_transforms = vbo_transforms.numpy()
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vbo_transforms.nbytes, vbo_transforms.ctypes.data, gl.GL_DYNAMIC_DRAW)

        # Create CUDA buffer for instance transforms
        self._instance_transform_cuda_buffer = wp.RegisteredGLBuffer(
            int(self.instance_transform_gl_buffer.value), self.device
        )

        self.update_colors(colors1, colors2)

        # Set up instance attribute pointers
        matrix_size = vbo_transforms[0].nbytes

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_transform_gl_buffer)

        # we can only send vec4s to the shader, so we need to split the instance transforms matrix into its column vectors
        for i in range(4):
            gl.glVertexAttribPointer(
                3 + i, 4, gl.GL_FLOAT, gl.GL_FALSE, matrix_size, ctypes.c_void_p(i * matrix_size // 4)
            )
            print(f"Allocating to attribute {3 + i}")
            gl.glEnableVertexAttribArray(3 + i)
            gl.glVertexAttribDivisor(3 + i, 1)

        gl.glBindVertexArray(0)

    def update_instances(self, transforms: wp.array = None, scalings: wp.array = None, colors1=None, colors2=None):
        from pyglet import gl

        if transforms is not None:
            if transforms.device.is_cuda:
                wp_transforms = transforms
            else:
                wp_transforms = transforms.to(self.device)
            self.transforms = wp_transforms
        if scalings is not None:
            if transforms.device.is_cuda:
                wp_scalings = scalings
            else:
                wp_scalings = scalings.to(self.device)
            self.scalings = wp_scalings

        if transforms is not None or scalings is not None:
            gl.glBindVertexArray(self.vao)
            vbo_transforms = self._instance_transform_cuda_buffer.map(dtype=wp.mat44, shape=(self.num_instances,))

            wp.launch(
                update_vbo_transforms,
                dim=self.num_instances,
                inputs=[
                    self.instance_ids,
                    None,
                    self.instance_transforms,
                    self.instance_scalings,
                    None,
                ],
                outputs=[
                    vbo_transforms,
                ],
                device=self.device,
            )

            self._instance_transform_cuda_buffer.unmap()

        if colors1 is not None or colors2 is not None:
            self.update_colors(colors1, colors2)

    def render(self):
        from pyglet import gl

        gl.glUseProgram(self.shape_shader.id)

        gl.glBindVertexArray(self.vao)
        gl.glDrawElementsInstanced(gl.GL_TRIANGLES, self.face_count, gl.GL_UNSIGNED_INT, None, self.num_instances)
        gl.glBindVertexArray(0)

    # scope exposes VBO transforms to be set directly by a warp kernel
    def __enter__(self):
        from pyglet import gl

        gl.glBindVertexArray(self.vao)
        self.vbo_transforms = self._instance_transform_cuda_buffer.map(dtype=wp.mat44, shape=(self.num_instances,))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._instance_transform_cuda_buffer.unmap()


def str_buffer(string: str):
    return ctypes.c_char_p(string.encode("utf-8"))


def arr_pointer(arr: np.ndarray):
    return arr.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))


class OpenGLRenderer:
    """
    OpenGLRenderer is a simple OpenGL renderer for rendering 3D shapes and meshes.
    """

    # number of segments to use for rendering spheres, capsules, cones and cylinders
    default_num_segments = 32

    def __init__(
        self,
        render_wireframe=False,
        enable_backface_culling=True,
    ):
        try:
            import pyglet

            # disable error checking for performance
            pyglet.options["debug_gl"] = False

            from pyglet import gl
            from pyglet.graphics.shader import Shader, ShaderProgram
            from pyglet.math import Vec3 as PyVec3
        except ImportError as e:
            raise Exception("OpenGLRenderer requires pyglet (version >= 2.0) to be installed.") from e


        self.render_wireframe = render_wireframe
        self.enable_backface_culling = enable_backface_culling

        self._device = wp.get_preferred_device()

        self._body_name = {}
        self._shapes = []
        self._shape_geo_hash = {}
        self._shape_gl_buffers = {}
        self._shape_instances = defaultdict(list)
        self._instances = {}
        self._instance_custom_ids = {}
        self._instance_shape = {}
        self._instance_gl_buffers = {}
        self._instance_transform_gl_buffer = None
        self._instance_transform_cuda_buffer = None
        self._instance_color1_buffer = None
        self._instance_color2_buffer = None
        self._instance_count = 0
        self._wp_instance_ids = None
        self._wp_instance_custom_ids = None
        self._np_instance_visible = None
        self._instance_ids = None
        self._inverse_instance_ids = None
        self._wp_instance_transforms = None
        self._wp_instance_scalings = None
        self._wp_instance_bodies = None
        self._update_shape_instances = False
        self._add_shape_instances = False

        # additional shape instancer used for points and line rendering
        self._shape_instancers = {}

        # instancer for the arrow shapes sof the coordinate system axes
        self._axis_instancer = None

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(True)
        gl.glDepthRange(0.0, 1.0)

        # self._shape_shader = get_default_shader()
        self._sun_direction = np.array((-0.2, 0.8, 0.3))
        self._sun_direction /= np.linalg.norm(self._sun_direction)
        self._shape_shader = ShaderProgram(
            Shader(shape_vertex_shader, "vertex"), Shader(shape_fragment_shader, "fragment")
        )
        # with self._shape_shader:
        #     gl.glUniform3f(
        #         gl.glGetUniformLocation(self._shape_shader.id, str_buffer("sunDirection")), *self._sun_direction
        #     )
        #     gl.glUniform3f(gl.glGetUniformLocation(self._shape_shader.id, str_buffer("lightColor")), 1, 1, 1)
            # self._loc_shape_model = gl.glGetUniformLocation(self._shape_shader.id, str_buffer("model"))
            # self._loc_shape_view = gl.glGetUniformLocation(self._shape_shader.id, str_buffer("view"))
            # self._loc_shape_projection = gl.glGetUniformLocation(self._shape_shader.id, str_buffer("projection"))
            # self._loc_shape_view_pos = gl.glGetUniformLocation(self._shape_shader.id, str_buffer("viewPos"))
            # gl.glUniform3f(self._loc_shape_view_pos, 0, 0, 10)


        check_gl_error()


    def clear(self):
        from pyglet import gl

        if self._instance_transform_gl_buffer is not None:
            try:
                gl.glDeleteBuffers(1, self._instance_transform_gl_buffer)
                gl.glDeleteBuffers(1, self._instance_color1_buffer)
                gl.glDeleteBuffers(1, self._instance_color2_buffer)
            except gl.GLException:
                pass
        for vao, vbo, ebo, _, _vertex_cuda_buffer in self._shape_gl_buffers.values():
            try:
                gl.glDeleteVertexArrays(1, vao)
                gl.glDeleteBuffers(1, vbo)
                gl.glDeleteBuffers(1, ebo)
            except gl.GLException:
                pass

        self._body_name.clear()
        self._shapes.clear()
        self._shape_geo_hash.clear()
        self._shape_gl_buffers.clear()
        self._shape_instances.clear()
        self._instances.clear()
        self._instance_shape.clear()
        self._instance_gl_buffers.clear()
        self._instance_transform_gl_buffer = None
        self._instance_transform_cuda_buffer = None
        self._instance_color1_buffer = None
        self._instance_color2_buffer = None
        self._wp_instance_ids = None
        self._wp_instance_custom_ids = None
        self._wp_instance_transforms = None
        self._wp_instance_scalings = None
        self._wp_instance_bodies = None
        self._np_instance_visible = None
        self._update_shape_instances = False

    @property
    def num_shapes(self):
        return len(self._shapes)

    @property
    def num_instances(self):
        return self._instance_count

    def draw(self):
        from pyglet import gl

        if self._add_shape_instances:
            self.allocate_shape_instances()
        if self._update_shape_instances:
            self.update_shape_instances()

        if self.enable_backface_culling:
            gl.glEnable(gl.GL_CULL_FACE)
        else:
            gl.glDisable(gl.GL_CULL_FACE)

        gl.glUseProgram(self._shape_shader.id)
        # view_mat_ptr = arr_pointer(self._view_matrix)
        # projection_mat_ptr = arr_pointer(self._projection_matrix)
        # gl.glUseProgram(self._shape_shader.id)
        # gl.glUniformMatrix4fv(self._loc_shape_view, 1, gl.GL_FALSE, view_mat_ptr)
        # gl.glUniform3f(self._loc_shape_view_pos, *self._camera_pos)
        # gl.glUniformMatrix4fv(self._loc_shape_projection, 1, gl.GL_FALSE, projection_mat_ptr)

        if self.render_wireframe:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)


        self._render_scene()

        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

    def _render_scene(self):
        from pyglet import gl

        start_instance_idx = 0

        for shape, (vao, _, _, tri_count, _) in self._shape_gl_buffers.items():
            num_instances = len(self._shape_instances[shape])

            gl.glBindVertexArray(vao)
            gl.glDrawElementsInstancedBaseInstance(
                gl.GL_TRIANGLES, tri_count, gl.GL_UNSIGNED_INT, None, num_instances, start_instance_idx
            )

            start_instance_idx += num_instances

        for instancer in self._shape_instancers.values():
            instancer.render()

        gl.glBindVertexArray(0)


    def register_shape(self, geo_hash, vertices, indices, color1=None, color2=None):
        from pyglet import gl

        shape = len(self._shapes)
        if color1 is None:
            color1 = tab10_color_map(len(self._shape_geo_hash))
        if color2 is None:
            color2 = np.clip(np.array(color1) + 0.25, 0.0, 1.0)
        # TODO check if we actually need to store the shape data
        self._shapes.append((vertices, indices, color1, color2, geo_hash))
        self._shape_geo_hash[geo_hash] = shape

        gl.glUseProgram(self._shape_shader.id)

        # Create VAO, VBO, and EBO
        vao = gl.GLuint()
        gl.glGenVertexArrays(1, vao)
        gl.glBindVertexArray(vao)

        vbo = gl.GLuint()
        gl.glGenBuffers(1, vbo)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices.ctypes.data, gl.GL_STATIC_DRAW)

        vertex_cuda_buffer = wp.RegisteredGLBuffer(int(vbo.value), self._device)

        ebo = gl.GLuint()
        gl.glGenBuffers(1, ebo)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices.ctypes.data, gl.GL_STATIC_DRAW)

        # Set up vertex attributes
        vertex_stride = vertices.shape[1] * vertices.itemsize
        # positions
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, vertex_stride, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        # normals
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, vertex_stride, ctypes.c_void_p(3 * vertices.itemsize))
        gl.glEnableVertexAttribArray(1)
        # uv coordinates
        gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, vertex_stride, ctypes.c_void_p(6 * vertices.itemsize))
        gl.glEnableVertexAttribArray(2)

        gl.glBindVertexArray(0)

        self._shape_gl_buffers[shape] = (vao, vbo, ebo, len(indices), vertex_cuda_buffer)

        return shape

    def deregister_shape(self, shape):
        from pyglet import gl

        if shape not in self._shape_gl_buffers:
            return

        vao, vbo, ebo, _, vertex_cuda_buffer = self._shape_gl_buffers[shape]
        try:
            gl.glDeleteVertexArrays(1, vao)
            gl.glDeleteBuffers(1, vbo)
            gl.glDeleteBuffers(1, ebo)
        except gl.GLException:
            pass

        _, _, _, _, geo_hash = self._shapes[shape]
        del self._shape_geo_hash[geo_hash]
        del self._shape_gl_buffers[shape]
        self._shapes.pop(shape)

    def add_shape_instance(
        self,
        name: str,
        shape: int,
        body,
        pos,
        rot,
        scale=(1.0, 1.0, 1.0),
        color1=None,
        color2=None,
        custom_index: int = -1,
        visible: bool = True,
    ):
        if color1 is None:
            color1 = self._shapes[shape][2]
        if color2 is None:
            color2 = self._shapes[shape][3]
        instance = len(self._instances)
        self._shape_instances[shape].append(instance)
        body = self._resolve_body_id(body)
        self._instances[name] = (instance, body, shape, [*pos, *rot], scale, color1, color2, visible)
        self._instance_shape[instance] = shape
        self._instance_custom_ids[instance] = custom_index
        self._add_shape_instances = True
        self._instance_count = len(self._instances)
        return instance

    def remove_shape_instance(self, name: str):
        if name not in self._instances:
            return

        instance, _, shape, _, _, _, _, _ = self._instances[name]

        self._shape_instances[shape].remove(instance)
        self._instance_count = len(self._instances)
        self._add_shape_instances = self._instance_count > 0
        del self._instance_shape[instance]
        del self._instance_custom_ids[instance]
        del self._instances[name]

    def update_instance_colors(self):
        from pyglet import gl

        colors1, colors2 = [], []
        all_instances = list(self._instances.values())
        for _shape, instances in self._shape_instances.items():
            for i in instances:
                if i >= len(all_instances):
                    continue
                instance = all_instances[i]
                colors1.append(instance[5])
                colors2.append(instance[6])
        colors1 = np.array(colors1, dtype=np.float32)
        colors2 = np.array(colors2, dtype=np.float32)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._instance_color1_buffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, colors1.nbytes, colors1.ctypes.data, gl.GL_STATIC_DRAW)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._instance_color2_buffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, colors2.nbytes, colors2.ctypes.data, gl.GL_STATIC_DRAW)

    def allocate_shape_instances(self):
        from pyglet import gl

        self._add_shape_instances = False
        self._wp_instance_transforms = wp.array(
            [instance[3] for instance in self._instances.values()], dtype=wp.transform, device=self._device
        )
        self._wp_instance_scalings = wp.array(
            [instance[4] for instance in self._instances.values()], dtype=wp.vec3, device=self._device
        )
        self._wp_instance_bodies = wp.array(
            [instance[1] for instance in self._instances.values()], dtype=wp.int32, device=self._device
        )

        gl.glUseProgram(self._shape_shader.id)
        if self._instance_transform_gl_buffer is not None:
            gl.glDeleteBuffers(1, self._instance_transform_gl_buffer)
            gl.glDeleteBuffers(1, self._instance_color1_buffer)
            gl.glDeleteBuffers(1, self._instance_color2_buffer)

        # create instance buffer and bind it as an instanced array
        self._instance_transform_gl_buffer = gl.GLuint()
        gl.glGenBuffers(1, self._instance_transform_gl_buffer)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._instance_transform_gl_buffer)

        transforms = np.tile(np.diag(np.ones(4, dtype=np.float32)), (len(self._instances), 1, 1))
        gl.glBufferData(gl.GL_ARRAY_BUFFER, transforms.nbytes, transforms.ctypes.data, gl.GL_DYNAMIC_DRAW)

        # create CUDA buffer for instance transforms
        self._instance_transform_cuda_buffer = wp.RegisteredGLBuffer(
            int(self._instance_transform_gl_buffer.value), self._device
        )

        # create color buffers
        self._instance_color1_buffer = gl.GLuint()
        gl.glGenBuffers(1, self._instance_color1_buffer)
        self._instance_color2_buffer = gl.GLuint()
        gl.glGenBuffers(1, self._instance_color2_buffer)

        self.update_instance_colors()

        # set up instance attribute pointers
        matrix_size = transforms[0].nbytes

        instance_ids = []
        instance_custom_ids = []
        instance_visible = []
        instances = list(self._instances.values())
        inverse_instance_ids = {}
        instance_count = 0
        colors_size = np.zeros(3, dtype=np.float32).nbytes
        for shape, (vao, _vbo, _ebo, _tri_count, _vertex_cuda_buffer) in self._shape_gl_buffers.items():
            gl.glBindVertexArray(vao)

            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._instance_transform_gl_buffer)

            # we can only send vec4s to the shader, so we need to split the instance transforms matrix into its column vectors
            for i in range(4):
                gl.glVertexAttribPointer(
                    3 + i, 4, gl.GL_FLOAT, gl.GL_FALSE, matrix_size, ctypes.c_void_p(i * matrix_size // 4)
                )
                gl.glEnableVertexAttribArray(3 + i)
                gl.glVertexAttribDivisor(3 + i, 1)

            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._instance_color1_buffer)
            gl.glVertexAttribPointer(7, 3, gl.GL_FLOAT, gl.GL_FALSE, colors_size, ctypes.c_void_p(0))
            gl.glEnableVertexAttribArray(7)
            gl.glVertexAttribDivisor(7, 1)

            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._instance_color2_buffer)
            gl.glVertexAttribPointer(8, 3, gl.GL_FLOAT, gl.GL_FALSE, colors_size, ctypes.c_void_p(0))
            gl.glEnableVertexAttribArray(8)
            gl.glVertexAttribDivisor(8, 1)

            instance_ids.extend(self._shape_instances[shape])
            for i in self._shape_instances[shape]:
                inverse_instance_ids[i] = instance_count
                instance_count += 1
                instance_custom_ids.append(self._instance_custom_ids[i])
                instance_visible.append(instances[i][7])

        # trigger update to the instance transforms
        self._update_shape_instances = True

        self._wp_instance_ids = wp.array(instance_ids, dtype=wp.int32, device=self._device)
        self._wp_instance_custom_ids = wp.array(instance_custom_ids, dtype=wp.int32, device=self._device)
        self._np_instance_visible = np.array(instance_visible)
        self._instance_ids = instance_ids
        self._inverse_instance_ids = inverse_instance_ids

        gl.glBindVertexArray(0)

    def update_shape_instance(self, name, pos=None, rot=None, color1=None, color2=None, visible=None):
        """Update the instance transform of the shape

        Args:
            name: The name of the shape
            pos: The position of the shape
            rot: The rotation of the shape
            color1: The first color of the checker pattern
            color2: The second color of the checker pattern
            visible: Whether the shape is visible
        """
        from pyglet import gl

        if name in self._instances:
            i, body, shape, tf, scale, old_color1, old_color2, v = self._instances[name]
            if visible is None:
                visible = v
            new_tf = np.copy(tf)
            if pos is not None:
                new_tf[:3] = pos
            if rot is not None:
                new_tf[3:] = rot
            self._instances[name] = (
                i,
                body,
                shape,
                new_tf,
                scale,
                old_color1 if color1 is None else color1,
                old_color2 if color2 is None else color2,
                visible,
            )
            self._update_shape_instances = True
            if color1 is not None or color2 is not None:
                vao, vbo, ebo, tri_count, vertex_cuda_buffer = self._shape_gl_buffers[shape]
                gl.glBindVertexArray(vao)
                self.update_instance_colors()
                gl.glBindVertexArray(0)
            return True
        return False

    def update_shape_instances(self):
        with self._shape_shader:
            self._update_shape_instances = False
            self._wp_instance_transforms = wp.array(
                [instance[3] for instance in self._instances.values()], dtype=wp.transform, device=self._device
            )
            self.update_body_transforms(None)

    def update_body_transforms(self, body_tf: wp.array):
        if self._instance_transform_cuda_buffer is None:
            return

        body_q = None
        if body_tf is not None:
            if body_tf.device.is_cuda:
                body_q = body_tf
            else:
                body_q = body_tf.to(self._device)

        vbo_transforms = self._instance_transform_cuda_buffer.map(dtype=wp.mat44, shape=(self._instance_count,))

        wp.launch(
            update_vbo_transforms,
            dim=self._instance_count,
            inputs=[
                self._wp_instance_ids,
                self._wp_instance_bodies,
                self._wp_instance_transforms,
                self._wp_instance_scalings,
                body_q,
            ],
            outputs=[
                vbo_transforms,
            ],
            device=self._device,
        )

        self._instance_transform_cuda_buffer.unmap()

    def register_body(self, name):
        # register body name and return its ID
        if name not in self._body_name:
            self._body_name[name] = len(self._body_name)
        return self._body_name[name]

    def _resolve_body_id(self, body):
        if body is None:
            return -1
        if isinstance(body, int):
            return body
        return self._body_name[body]

    def render_plane(
        self,
        name: str,
        pos: tuple,
        rot: tuple,
        width: float,
        length: float,
        color: tuple = (1.0, 1.0, 1.0),
        color2=None,
        parent_body: str = None,
        is_template: bool = False,
        u_scaling=1.0,
        v_scaling=1.0,
    ):
        """Add a plane for visualization

        Args:
            name: The name of the plane
            pos: The position of the plane
            rot: The rotation of the plane
            width: The width of the plane
            length: The length of the plane
            color: The color of the plane
            texture: The texture of the plane (optional)
        """
        geo_hash = hash(("plane", width, length))
        if geo_hash in self._shape_geo_hash:
            shape = self._shape_geo_hash[geo_hash]
            if self.update_shape_instance(name, pos, rot):
                return shape
        else:
            faces = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
            normal = (0.0, 1.0, 0.0)
            width = width if width > 0.0 else 100.0
            length = length if length > 0.0 else 100.0
            aspect = width / length
            u = width * aspect * u_scaling
            v = length * v_scaling
            gfx_vertices = np.array(
                [
                    [-width, 0.0, -length, *normal, 0.0, 0.0],
                    [-width, 0.0, length, *normal, 0.0, v],
                    [width, 0.0, length, *normal, u, v],
                    [width, 0.0, -length, *normal, u, 0.0],
                ],
                dtype=np.float32,
            )
            shape = self.register_shape(geo_hash, gfx_vertices, faces, color1=color, color2=color2)
        if not is_template:
            body = self._resolve_body_id(parent_body)
            self.add_shape_instance(name, shape, body, pos, rot)
        return shape

    def render_ground(self, size: float = 1000.0, plane=None):
        """Add a ground plane for visualization

        Args:
            size: The size of the ground plane
        """
        color1 = (200 / 255, 200 / 255, 200 / 255)
        color2 = (150 / 255, 150 / 255, 150 / 255)
        sqh = np.sqrt(0.5)
        # if self._camera_axis == 0:
        #     q = (0.0, 0.0, -sqh, sqh)
        # elif self._camera_axis == 1:
        #     q = (0.0, 0.0, 0.0, 1.0)
        # elif self._camera_axis == 2:
            # q = (sqh, 0.0, 0.0, sqh)
        q = (sqh, 0.0, 0.0, sqh)
        pos = (0.0, 0.0, 0.0)
        if plane is not None:
            normal = np.array(plane[:3])
            normal /= np.linalg.norm(normal)
            pos = plane[3] * normal
            if np.allclose(normal, (0.0, 1.0, 0.0)):
                # no rotation necessary
                q = (0.0, 0.0, 0.0, 1.0)
            else:
                c = np.cross(normal, (0.0, 1.0, 0.0))
                angle = np.arcsin(np.linalg.norm(c))
                axis = np.abs(c) / np.linalg.norm(c)
                q = wp.quat_from_axis_angle(axis, angle)
        return self.render_plane(
            "ground",
            pos,
            q,
            size,
            size,
            color1,
            color2=color2,
            u_scaling=1.0,
            v_scaling=1.0,
        )

    def render_sphere(
        self,
        name: str,
        pos: tuple,
        rot: tuple,
        radius: float,
        parent_body: str = None,
        is_template: bool = False,
        color=None,
    ):
        """Add a sphere for visualization

        Args:
            pos: The position of the sphere
            radius: The radius of the sphere
            name: A name for the USD prim on the stage
            color: The color of the sphere
        """
        geo_hash = hash(("sphere", radius))
        if geo_hash in self._shape_geo_hash:
            shape = self._shape_geo_hash[geo_hash]
            if self.update_shape_instance(name, pos, rot, color1=color, color2=color):
                return shape
        else:
            vertices, indices = self._create_sphere_mesh(radius)
            shape = self.register_shape(geo_hash, vertices, indices, color1=color, color2=color)
        if not is_template:
            body = self._resolve_body_id(parent_body)
            self.add_shape_instance(name, shape, body, pos, rot, color1=color, color2=color)
        return shape

    def render_capsule(
        self,
        name: str,
        pos: tuple,
        rot: tuple,
        radius: float,
        half_height: float,
        parent_body: str = None,
        is_template: bool = False,
        up_axis: int = 1,
        color: tuple = None,
    ):
        """Add a capsule for visualization

        Args:
            pos: The position of the capsule
            radius: The radius of the capsule
            half_height: The half height of the capsule
            name: A name for the USD prim on the stage
            up_axis: The axis of the capsule that points up (0: x, 1: y, 2: z)
            color: The color of the capsule
        """
        geo_hash = hash(("capsule", radius, half_height))
        if geo_hash in self._shape_geo_hash:
            shape = self._shape_geo_hash[geo_hash]
            if self.update_shape_instance(name, pos, rot):
                return shape
        else:
            vertices, indices = self._create_capsule_mesh(radius, half_height, up_axis=up_axis)
            shape = self.register_shape(geo_hash, vertices, indices, color1=color, color2=color)
        if not is_template:
            body = self._resolve_body_id(parent_body)
            self.add_shape_instance(name, shape, body, pos, rot)
        return shape

    def render_cylinder(
        self,
        name: str,
        pos: tuple,
        rot: tuple,
        radius: float,
        half_height: float,
        parent_body: str = None,
        is_template: bool = False,
        up_axis: int = 1,
        color: tuple = None,
    ):
        """Add a cylinder for visualization

        Args:
            pos: The position of the cylinder
            radius: The radius of the cylinder
            half_height: The half height of the cylinder
            name: A name for the USD prim on the stage
            up_axis: The axis of the cylinder that points up (0: x, 1: y, 2: z)
            color: The color of the capsule
        """
        geo_hash = hash(("cylinder", radius, half_height))
        if geo_hash in self._shape_geo_hash:
            shape = self._shape_geo_hash[geo_hash]
            if self.update_shape_instance(name, pos, rot):
                return shape
        else:
            vertices, indices = self._create_cylinder_mesh(radius, half_height, up_axis=up_axis)
            shape = self.register_shape(geo_hash, vertices, indices, color1=color, color2=color)
        if not is_template:
            body = self._resolve_body_id(parent_body)
            self.add_shape_instance(name, shape, body, pos, rot)
        return shape

    def render_cone(
        self,
        name: str,
        pos: tuple,
        rot: tuple,
        radius: float,
        half_height: float,
        parent_body: str = None,
        is_template: bool = False,
        up_axis: int = 1,
        color: tuple = None,
    ):
        """Add a cone for visualization

        Args:
            pos: The position of the cone
            radius: The radius of the cone
            half_height: The half height of the cone
            name: A name for the USD prim on the stage
            up_axis: The axis of the cone that points up (0: x, 1: y, 2: z)
            color: The color of the cone
        """
        geo_hash = hash(("cone", radius, half_height))
        if geo_hash in self._shape_geo_hash:
            shape = self._shape_geo_hash[geo_hash]
            if self.update_shape_instance(name, pos, rot):
                return shape
        else:
            vertices, indices = self._create_cone_mesh(radius, half_height, up_axis=up_axis)
            shape = self.register_shape(geo_hash, vertices, indices, color1=color, color2=color)
        if not is_template:
            body = self._resolve_body_id(parent_body)
            self.add_shape_instance(name, shape, body, pos, rot)
        return shape

    def render_box(
        self,
        name: str,
        pos: tuple,
        rot: tuple,
        extents: tuple,
        parent_body: str = None,
        is_template: bool = False,
        color: tuple = None,
    ):
        """Add a box for visualization

        Args:
            pos: The position of the box
            extents: The extents of the box
            name: A name for the USD prim on the stage
            color: The color of the box
        """
        geo_hash = hash(("box", tuple(extents)))
        if geo_hash in self._shape_geo_hash:
            shape = self._shape_geo_hash[geo_hash]
            if self.update_shape_instance(name, pos, rot):
                return shape
        else:
            vertices, indices = self._create_box_mesh(extents)
            shape = self.register_shape(geo_hash, vertices, indices, color1=color, color2=color)
        if not is_template:
            body = self._resolve_body_id(parent_body)
            self.add_shape_instance(name, shape, body, pos, rot)
        return shape

    def render_mesh(
        self,
        name: str,
        points,
        indices,
        colors=None,
        pos=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
        scale=(1.0, 1.0, 1.0),
        update_topology=False,
        parent_body: str = None,
        is_template: bool = False,
        smooth_shading: bool = True,
    ):
        """Add a mesh for visualization

        Args:
            points: The points of the mesh
            indices: The indices of the mesh
            colors: The colors of the mesh
            pos: The position of the mesh
            rot: The rotation of the mesh
            scale: The scale of the mesh
            name: A name for the USD prim on the stage
            smooth_shading: Whether to average face normals at each vertex or introduce additional vertices for each face
        """
        if colors is not None:
            colors = np.array(colors, dtype=np.float32)

        points = np.array(points, dtype=np.float32)
        point_count = len(points)

        indices = np.array(indices, dtype=np.int32).reshape((-1, 3))
        idx_count = len(indices)

        geo_hash = hash((indices.tobytes(),))

        if name in self._instances:
            # We've already registered this mesh instance and its associated shape.
            shape = self._instances[name][2]
        else:
            if geo_hash in self._shape_geo_hash:
                # We've only registered the shape, which can happen when `is_template` is `True`.
                shape = self._shape_geo_hash[geo_hash]
            else:
                shape = None

        # Check if we already have that shape registered and can perform
        # minimal updates since the topology is not changing, before exiting.
        if not update_topology:
            if name in self._instances:
                # Update the instance's transform.
                self.update_shape_instance(name, pos, rot, color1=colors)

            if shape is not None:
                # Update the shape's point positions.
                self.update_shape_vertices(shape, points, scale)
                return shape

        # No existing shape for the given mesh was found, or its topology may have changed,
        # so we need to define a new one either way.
        if smooth_shading:
            normals = wp.zeros(point_count, dtype=wp.vec3)
            vertices = wp.array(points, dtype=wp.vec3)
            faces_per_vertex = wp.zeros(point_count, dtype=int)
            wp.launch(
                compute_average_normals,
                dim=idx_count,
                inputs=[wp.array(indices, dtype=int), vertices, scale],
                outputs=[normals, faces_per_vertex],
            )
            gfx_vertices = wp.zeros((point_count, 8), dtype=float)
            wp.launch(
                assemble_gfx_vertices,
                dim=point_count,
                inputs=[vertices, normals, faces_per_vertex, scale],
                outputs=[gfx_vertices],
            )
            gfx_vertices = gfx_vertices.numpy()
            gfx_indices = indices.flatten()
        else:
            gfx_vertices = wp.zeros((idx_count * 3, 8), dtype=float)
            wp.launch(
                compute_gfx_vertices,
                dim=idx_count,
                inputs=[wp.array(indices, dtype=int), wp.array(points, dtype=wp.vec3), scale],
                outputs=[gfx_vertices],
            )
            gfx_vertices = gfx_vertices.numpy()
            gfx_indices = np.arange(idx_count * 3)

        # If there was a shape for the given mesh, clean it up.
        if shape is not None:
            self.deregister_shape(shape)

        # If there was an instance for the given mesh, clean it up.
        if name in self._instances:
            self.remove_shape_instance(name)

        # Register the new shape.
        shape = self.register_shape(geo_hash, gfx_vertices, gfx_indices)

        if not is_template:
            # Create a new instance if necessary.
            body = self._resolve_body_id(parent_body)
            self.add_shape_instance(name, shape, body, pos, rot, color1=colors)

        return shape

    def render_arrow(
        self,
        name: str,
        pos: tuple,
        rot: tuple,
        base_radius: float,
        base_height: float,
        cap_radius: float = None,
        cap_height: float = None,
        parent_body: str = None,
        is_template: bool = False,
        up_axis: int = 1,
        color: Tuple[float, float, float] = None,
    ):
        """Add a arrow for visualization

        Args:
            pos: The position of the arrow
            base_radius: The radius of the cylindrical base of the arrow
            base_height: The height of the cylindrical base of the arrow
            cap_radius: The radius of the conical cap of the arrow
            cap_height: The height of the conical cap of the arrow
            name: A name for the USD prim on the stage
            up_axis: The axis of the arrow that points up (0: x, 1: y, 2: z)
        """
        geo_hash = hash(("arrow", base_radius, base_height, cap_radius, cap_height))
        if geo_hash in self._shape_geo_hash:
            shape = self._shape_geo_hash[geo_hash]
            if self.update_shape_instance(name, pos, rot):
                return shape
        else:
            vertices, indices = self._create_arrow_mesh(
                base_radius, base_height, cap_radius, cap_height, up_axis=up_axis
            )
            shape = self.register_shape(geo_hash, vertices, indices)
        if not is_template:
            body = self._resolve_body_id(parent_body)
            self.add_shape_instance(name, shape, body, pos, rot, color1=color, color2=color)
        return shape

    def render_ref(self, name: str, path: str, pos: tuple, rot: tuple, scale: tuple, color: tuple = None):
        """
        Create a reference (instance) with the given name to the given path.
        """

        if path in self._instances:
            _, body, shape, _, original_scale, color1, color2 = self._instances[path]
            if color is not None:
                color1 = color2 = color
            self.add_shape_instance(name, shape, body, pos, rot, scale or original_scale, color1, color2)
            return

        raise Exception("Cannot create reference to path: " + path)

    def render_points(self, name: str, points, radius, colors=None):
        """Add a set of points

        Args:
            points: The points to render
            radius: The radius of the points (scalar or list)
            colors: The colors of the points
            name: A name for the USD prim on the stage
        """

        if len(points) == 0:
            return

        if isinstance(points, wp.array):
            wp_points = points
        else:
            wp_points = wp.array(points, dtype=wp.vec3, device=self._device)

        if name not in self._shape_instancers:
            np_points = points.numpy() if isinstance(points, wp.array) else points
            instancer = ShapeInstancer(self._shape_shader, self._device)
            radius_is_scalar = np.isscalar(radius)
            if radius_is_scalar:
                vertices, indices = self._create_sphere_mesh(radius)
            else:
                vertices, indices = self._create_sphere_mesh(1.0)
            if colors is None:
                color = tab10_color_map(len(self._shape_geo_hash))
            elif len(colors) == 3:
                color = colors
            else:
                color = colors[0]
            instancer.register_shape(vertices, indices, color, color)
            scalings = None if radius_is_scalar else np.tile(radius, (3, 1)).T
            instancer.allocate_instances(np_points, colors1=colors, colors2=colors, scalings=scalings)
            self._shape_instancers[name] = instancer
        else:
            instancer = self._shape_instancers[name]
            if len(points) != instancer.num_instances:
                np_points = points.numpy() if isinstance(points, wp.array) else points
                instancer.allocate_instances(np_points)

        with instancer:
            wp.launch(
                update_points_positions,
                dim=len(points),
                inputs=[wp_points, instancer.instance_scalings],
                outputs=[instancer.vbo_transforms],
                device=self._device,
            )

    def _render_lines(self, name: str, lines, color: tuple, radius: float = 0.01):
        if len(lines) == 0:
            return

        if name not in self._shape_instancers:
            instancer = ShapeInstancer(self._shape_shader, self._device)
            vertices, indices = self._create_capsule_mesh(radius, 0.5)
            if color is None or isinstance(color, list) and len(color) > 0 and isinstance(color[0], list):
                color = tab10_color_map(len(self._shape_geo_hash))
            instancer.register_shape(vertices, indices, color, color)
            instancer.allocate_instances(np.zeros((len(lines), 3)))
            self._shape_instancers[name] = instancer
        else:
            instancer = self._shape_instancers[name]
            if len(lines) != instancer.num_instances:
                instancer.allocate_instances(np.zeros((len(lines), 3)))
            instancer.update_colors(color, color)

        lines_wp = wp.array(lines, dtype=wp.vec3, ndim=2, device=self._device)
        with instancer:
            wp.launch(
                update_line_transforms,
                dim=len(lines),
                inputs=[lines_wp],
                outputs=[instancer.vbo_transforms],
                device=self._device,
            )

    def render_line_list(self, name: str, vertices, indices, color: tuple = None, radius: float = 0.01):
        """Add a line list as a set of capsules

        Args:
            vertices: The vertices of the line-list
            indices: The indices of the line-list
            color: The color of the line
            radius: The radius of the line
        """
        lines = []
        for i in range(len(indices) // 2):
            lines.append((vertices[indices[2 * i]], vertices[indices[2 * i + 1]]))
        lines = np.array(lines)
        self._render_lines(name, lines, color, radius)

    def render_line_strip(self, name: str, vertices, color: tuple = None, radius: float = 0.01):
        """Add a line strip as a set of capsules

        Args:
            vertices: The vertices of the line-strip
            color: The color of the line
            radius: The radius of the line
        """
        lines = []
        for i in range(len(vertices) - 1):
            lines.append((vertices[i], vertices[i + 1]))
        lines = np.array(lines)
        self._render_lines(name, lines, color, radius)

    def update_shape_vertices(self, shape, points, scale):
        if isinstance(points, wp.array):
            wp_points = points.to(self._device)
        else:
            wp_points = wp.array(points, dtype=wp.vec3, device=self._device)

        cuda_buffer = self._shape_gl_buffers[shape][4]
        vertices_shape = self._shapes[shape][0].shape
        vbo_vertices = cuda_buffer.map(dtype=wp.float32, shape=vertices_shape)

        wp.launch(
            update_vbo_vertices,
            dim=vertices_shape[0],
            inputs=[wp_points, scale],
            outputs=[vbo_vertices],
            device=self._device,
        )

        cuda_buffer.unmap()

    @staticmethod
    def _create_sphere_mesh(
        radius=1.0,
        num_latitudes=default_num_segments,
        num_longitudes=default_num_segments,
        reverse_winding=False,
    ):
        vertices = []
        indices = []

        for i in range(num_latitudes + 1):
            theta = i * np.pi / num_latitudes
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)

            for j in range(num_longitudes + 1):
                phi = j * 2 * np.pi / num_longitudes
                sin_phi = np.sin(phi)
                cos_phi = np.cos(phi)

                x = cos_phi * sin_theta
                y = cos_theta
                z = sin_phi * sin_theta

                u = float(j) / num_longitudes
                v = float(i) / num_latitudes

                vertices.append([x * radius, y * radius, z * radius, x, y, z, u, v])

        for i in range(num_latitudes):
            for j in range(num_longitudes):
                first = i * (num_longitudes + 1) + j
                second = first + num_longitudes + 1

                if reverse_winding:
                    indices.extend([first, second, first + 1, second, second + 1, first + 1])
                else:
                    indices.extend([first, first + 1, second, second, first + 1, second + 1])

        return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)

    @staticmethod
    def _create_capsule_mesh(radius, half_height, up_axis=1, segments=default_num_segments):
        vertices = []
        indices = []

        x_dir, y_dir, z_dir = ((1, 2, 0), (2, 0, 1), (0, 1, 2))[up_axis]
        up_vector = np.zeros(3)
        up_vector[up_axis] = half_height

        for i in range(segments + 1):
            theta = i * np.pi / segments
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)

            for j in range(segments + 1):
                phi = j * 2 * np.pi / segments
                sin_phi = np.sin(phi)
                cos_phi = np.cos(phi)

                z = cos_phi * sin_theta
                y = cos_theta
                x = sin_phi * sin_theta

                u = cos_theta * 0.5 + 0.5
                v = cos_phi * sin_theta * 0.5 + 0.5

                xyz = x, y, z
                x, y, z = xyz[x_dir], xyz[y_dir], xyz[z_dir]
                xyz = np.array((x, y, z), dtype=np.float32) * radius
                if j < segments // 2:
                    xyz += up_vector
                else:
                    xyz -= up_vector

                vertices.append([*xyz, x, y, z, u, v])

        nv = len(vertices)
        for i in range(segments + 1):
            for j in range(segments + 1):
                first = (i * (segments + 1) + j) % nv
                second = (first + segments + 1) % nv
                indices.extend([first, second, (first + 1) % nv, second, (second + 1) % nv, (first + 1) % nv])

        vertex_data = np.array(vertices, dtype=np.float32)
        index_data = np.array(indices, dtype=np.uint32)

        return vertex_data, index_data

    @staticmethod
    def _create_cone_mesh(radius, half_height, up_axis=1, segments=default_num_segments):
        # render it as a cylinder with zero top radius so we get correct normals on the sides
        return OpenGLRenderer._create_cylinder_mesh(radius, half_height, up_axis, segments, 0.0)

    @staticmethod
    def _create_cylinder_mesh(radius, half_height, up_axis=1, segments=default_num_segments, top_radius=None):
        if up_axis not in (0, 1, 2):
            raise ValueError("up_axis must be between 0 and 2")

        x_dir, y_dir, z_dir = (
            (1, 2, 0),
            (0, 1, 2),
            (2, 0, 1),
        )[up_axis]

        indices = []

        cap_vertices = []
        side_vertices = []

        # create center cap vertices
        position = np.array([0, -half_height, 0])[[x_dir, y_dir, z_dir]]
        normal = np.array([0, -1, 0])[[x_dir, y_dir, z_dir]]
        cap_vertices.append([*position, *normal, 0.5, 0.5])
        cap_vertices.append([*-position, *-normal, 0.5, 0.5])

        if top_radius is None:
            top_radius = radius
        side_slope = -np.arctan2(top_radius - radius, 2 * half_height)

        # create the cylinder base and top vertices
        for j in (-1, 1):
            center_index = max(j, 0)
            if j == 1:
                radius = top_radius
            for i in range(segments):
                theta = 2 * np.pi * i / segments

                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)

                x = cos_theta
                y = j * half_height
                z = sin_theta

                position = np.array([radius * x, y, radius * z])

                normal = np.array([x, side_slope, z])
                normal = normal / np.linalg.norm(normal)
                uv = (i / (segments - 1), (j + 1) / 2)
                vertex = np.hstack([position[[x_dir, y_dir, z_dir]], normal[[x_dir, y_dir, z_dir]], uv])
                side_vertices.append(vertex)

                normal = np.array([0, j, 0])
                uv = (cos_theta * 0.5 + 0.5, sin_theta * 0.5 + 0.5)
                vertex = np.hstack([position[[x_dir, y_dir, z_dir]], normal[[x_dir, y_dir, z_dir]], uv])
                cap_vertices.append(vertex)

                cs = center_index * segments
                indices.extend([center_index, i + cs + 2, (i + 1) % segments + cs + 2][::-j])

        # create the cylinder side indices
        for i in range(segments):
            index1 = len(cap_vertices) + i + segments
            index2 = len(cap_vertices) + ((i + 1) % segments) + segments
            index3 = len(cap_vertices) + i
            index4 = len(cap_vertices) + ((i + 1) % segments)

            indices.extend([index1, index2, index3, index2, index4, index3])

        vertex_data = np.array(np.vstack((cap_vertices, side_vertices)), dtype=np.float32)
        index_data = np.array(indices, dtype=np.uint32)

        return vertex_data, index_data

    @staticmethod
    def _create_arrow_mesh(
        base_radius, base_height, cap_radius=None, cap_height=None, up_axis=1, segments=default_num_segments
    ):
        if up_axis not in (0, 1, 2):
            raise ValueError("up_axis must be between 0 and 2")
        if cap_radius is None:
            cap_radius = base_radius * 1.8
        if cap_height is None:
            cap_height = base_height * 0.18

        up_vector = np.array([0, 0, 0])
        up_vector[up_axis] = 1

        base_vertices, base_indices = OpenGLRenderer._create_cylinder_mesh(
            base_radius, base_height / 2, up_axis, segments
        )
        cap_vertices, cap_indices = OpenGLRenderer._create_cone_mesh(cap_radius, cap_height / 2, up_axis, segments)

        base_vertices[:, :3] += base_height / 2 * up_vector
        # move cap slightly lower to avoid z-fighting
        cap_vertices[:, :3] += (base_height + cap_height / 2 - 1e-3 * base_height) * up_vector

        vertex_data = np.vstack((base_vertices, cap_vertices))
        index_data = np.hstack((base_indices, cap_indices + len(base_vertices)))

        return vertex_data, index_data

    @staticmethod
    def _create_box_mesh(extents):
        x_extent, y_extent, z_extent = extents

        vertices = [
            # Position                        Normal    UV
            [-x_extent, -y_extent, -z_extent, -1, 0, 0, 0, 0],
            [-x_extent, -y_extent, z_extent, -1, 0, 0, 1, 0],
            [-x_extent, y_extent, z_extent, -1, 0, 0, 1, 1],
            [-x_extent, y_extent, -z_extent, -1, 0, 0, 0, 1],
            [x_extent, -y_extent, -z_extent, 1, 0, 0, 0, 0],
            [x_extent, -y_extent, z_extent, 1, 0, 0, 1, 0],
            [x_extent, y_extent, z_extent, 1, 0, 0, 1, 1],
            [x_extent, y_extent, -z_extent, 1, 0, 0, 0, 1],
            [-x_extent, -y_extent, -z_extent, 0, -1, 0, 0, 0],
            [-x_extent, -y_extent, z_extent, 0, -1, 0, 1, 0],
            [x_extent, -y_extent, z_extent, 0, -1, 0, 1, 1],
            [x_extent, -y_extent, -z_extent, 0, -1, 0, 0, 1],
            [-x_extent, y_extent, -z_extent, 0, 1, 0, 0, 0],
            [-x_extent, y_extent, z_extent, 0, 1, 0, 1, 0],
            [x_extent, y_extent, z_extent, 0, 1, 0, 1, 1],
            [x_extent, y_extent, -z_extent, 0, 1, 0, 0, 1],
            [-x_extent, -y_extent, -z_extent, 0, 0, -1, 0, 0],
            [-x_extent, y_extent, -z_extent, 0, 0, -1, 1, 0],
            [x_extent, y_extent, -z_extent, 0, 0, -1, 1, 1],
            [x_extent, -y_extent, -z_extent, 0, 0, -1, 0, 1],
            [-x_extent, -y_extent, z_extent, 0, 0, 1, 0, 0],
            [-x_extent, y_extent, z_extent, 0, 0, 1, 1, 0],
            [x_extent, y_extent, z_extent, 0, 0, 1, 1, 1],
            [x_extent, -y_extent, z_extent, 0, 0, 1, 0, 1],
        ]

        # fmt: off
        indices = [
            0, 1, 2,
            0, 2, 3,
            4, 6, 5,
            4, 7, 6,
            8, 10, 9,
            8, 11, 10,
            12, 13, 14,
            12, 14, 15,
            16, 17, 18,
            16, 18, 19,
            20, 22, 21,
            20, 23, 22,
        ]
        # fmt: on
        return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)

class OpenGLRendererWrapper(OpenGLRenderer):
    def __init__(self, path, scaling=1.0, fps=60.0, up_axis=(0.0, 0.0, 1.0)):
        super().__init__()

