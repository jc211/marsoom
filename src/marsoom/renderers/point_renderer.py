import ctypes
from typing import Optional

import warp as wp
import torch
from pyglet import gl
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.graphics import vertexbuffer, vertexarray

from marsoom.context_3d import Context3D

point_vertex_shader = """
#version 330 core
uniform mat4 world2proj;

layout (location = 0) in vec3 pos_;
layout (location = 1) in vec4 color_;

out vec4 color;

void main() {
    color = color_;
    gl_Position = world2proj * vec4(pos_, 1.0);
}

"""

point_fragment_shader = """
#version 330 core
in vec4 color;
out vec4 FragColor;
void main()
{
    FragColor = color;
}
"""

class PointRenderer:
    def __init__(self):

        self.program = ShaderProgram(
            Shader(point_vertex_shader, "vertex"),
            Shader(point_fragment_shader, "fragment"),
        )
        self.num_floats_per_element = [3, 4]  # position, color
        self.total_floats_per_element = sum(self.num_floats_per_element)
        self._generate_gl_objects()

    def _generate_gl_objects(self):
        self.vao = vertexarray.VertexArray()
        self.vbo = vertexbuffer.BufferObject(size=1)
        self.gl_buffer = wp.RegisteredGLBuffer(
            self.vbo.id, wp.get_cuda_device(), flags=wp.RegisteredGLBuffer.WRITE_DISCARD
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

    def _resize(self, num_points: int):
        if num_points == self._num_points:
            return
        self._num_points = num_points
        struct_size = self.total_floats_per_element * ctypes.sizeof(ctypes.c_float)
        self.vbo.resize(num_points * struct_size)

    def update(self, positions: torch.Tensor, colors: Optional[torch.Tensor] = None):
        assert isinstance(positions, torch.Tensor) and positions.shape[1] == 3
        if colors is None:
            colors = torch.rand(positions.shape[0], 4, device=positions.device)
        assert isinstance(colors, torch.Tensor) and colors.shape[1] == 4 
        if positions.device != torch.device("cuda"):
            positions = positions.cuda()
        new_num_points = positions.shape[0]
        self._resize(new_num_points)

        temp = self.gl_buffer.map(
            dtype=wp.float32, shape=(self._num_points, self.total_floats_per_element)
        )
        temp_torch = wp.to_torch(temp)
        with torch.no_grad():
            temp_torch[:, :3] = positions
            temp_torch[:, 3:7] = colors
        self.gl_buffer.unmap()

    def draw(
        self,
        context: Context3D,
        point_size: float = 1.0,
        start_index: int = 0,
    ):
        gl.glPointSize(point_size)
        self.program.use()
        self.vao.bind()
        self.program["world2proj"] = context.world2projT
        gl.glDrawArrays(gl.GL_POINTS, start_index, self._num_points)
        self.program.stop()
        self.vao.unbind()
