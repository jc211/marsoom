import ctypes
import warp as wp
import numpy as np
import torch

from pyglet import gl
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.graphics import vertexbuffer, vertexarray

vertex_shader = """
#version 330 core
uniform mat4 world2proj;
uniform mat4 world;
layout (location = 0) in vec3 pos_;
void main() {
    gl_Position = world2proj * vec4(pos_, 1);
}
"""

fragment_shader = """
#version 330 core
out vec4 FragColor;

uniform vec3 color;

void main()
{
    FragColor = vec4(color, 1.0); 
}
"""


class SpringRenderer:
    def __init__(self):

        self.program = ShaderProgram(
            Shader(vertex_shader, "vertex"),
            Shader(fragment_shader, "fragment"),
        )
        self.num_floats_per_element = [3]  # position, quat
        self.total_floats_per_element = sum(self.num_floats_per_element)
        self._generate_gl_objects()

    def _generate_gl_objects(self):

        self.vao = vertexarray.VertexArray()
        self.vbo = vertexbuffer.BufferObject(size=1)
        self.gl_buffer = wp.RegisteredGLBuffer(
            self.vbo.id, wp.get_cuda_device(), flags=wp.RegisteredGLBuffer.WRITE_DISCARD
        )

        self.mesh_ebo = vertexbuffer.BufferObject(size=1, usage=gl.GL_STATIC_DRAW)
        self.mesh_ebo_buffer = wp.RegisteredGLBuffer(
            self.mesh_ebo.id,
            wp.get_cuda_device(),
            flags=wp.RegisteredGLBuffer.WRITE_DISCARD,
        )

        self.program.use()
        self.vao.bind()

        self.vbo.bind()

        offset = 0
        stride = self.total_floats_per_element * ctypes.sizeof(ctypes.c_float)
        for i, num_floats in enumerate(self.num_floats_per_element):
            gl.glEnableVertexAttribArray(i)
            gl.glVertexAttribPointer(
                i, num_floats, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(offset)
            )
            offset += num_floats * ctypes.sizeof(ctypes.c_float)

        self.vao.unbind()
        self.program.stop()
        self._num_points = 0
        self._num_lines = 0

    def _resize(self, num_points: int, num_lines: int):
        self._num_points = num_points
        self._num_lines = num_lines
        struct_size = self.total_floats_per_element * ctypes.sizeof(ctypes.c_float)
        self.vbo.resize(num_points * struct_size)
        line_size = 2 * ctypes.sizeof(ctypes.c_uint32)
        self.mesh_ebo.resize(num_lines * line_size)

    def _copy(self, positions: torch.Tensor, indices: torch.Tensor):
        temp = self.gl_buffer.map(
            dtype=wp.float32, shape=(self._num_points, self.total_floats_per_element)
        )
        temp_torch = wp.to_torch(temp)
        with torch.no_grad():
            temp_torch[:, :3] = positions
        self.gl_buffer.unmap()

        temp = self.mesh_ebo_buffer.map(dtype=wp.uint32, shape=(indices.shape[0] * 2,))
        indices_wp = wp.from_torch(indices, dtype=wp.uint32)
        wp.copy(temp, indices_wp)
        self.mesh_ebo_buffer.unmap()

    def update(
        self,
        positions: torch.Tensor,
        indices: torch.Tensor,
    ):
        assert positions.shape[1] == 3
        assert indices.shape[1] == 2
        assert positions.device == indices.device
        assert positions.device.type == "cuda"
        new_num_points = positions.shape[0]
        num_new_lines = indices.shape[0]
        if new_num_points != self._num_points or num_new_lines != self._num_lines:
            self._resize(new_num_points, num_new_lines)
        self._copy(positions, indices)

    def draw(
        self,
        world2projT,
        model=np.eye(4, dtype=np.float32),
        line_width: float = 1.0,
        color: np.ndarray = np.array([0.078, 0.129, 0.239], dtype=np.float32),
    ):

        self.program.use()
        self.vao.bind()
        # bind ebo
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.mesh_ebo.id)

        gl.glLineWidth(line_width)
        self.program["world2proj"] = world2projT
        self.program["color"] = color
        gl.glDrawElements(gl.GL_LINES, self._num_lines * 2, gl.GL_UNSIGNED_INT, 0)
        self.program.stop()
        self.vao.unbind()
