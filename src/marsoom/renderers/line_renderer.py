import ctypes
from typing import Tuple

import warp as wp
import torch

from pyglet import gl
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.graphics import vertexbuffer, vertexarray

vertex_shader = """
#version 330 core
uniform mat4 world2proj;

layout (location = 0) in vec3 pos_;

void main() {
    gl_Position = world2proj * vec4(pos_, 1);
}
"""

fragment_shader = """
#version 330 core
uniform vec3 color;
out vec4 FragColor;
void main()
{
    FragColor = vec4(color, 1.0); 
}
"""


class LineRenderer:
    def __init__(self):
        self.program = ShaderProgram(
            Shader(vertex_shader, "vertex"),
            Shader(fragment_shader, "fragment"),
        )
        self.vao = vertexarray.VertexArray()
        self.vbo = vertexbuffer.BufferObject(size=1)
        self.gl_buffer = wp.RegisteredGLBuffer(
            self.vbo.id, wp.get_cuda_device(), flags=wp.RegisteredGLBuffer.WRITE_DISCARD
        )
        self._size = 0
        self._resize()

    def _resize(self, num: int = 1):
        if num == self._size:
            return

        self.program.use()
        self.vao.bind()
        self.vbo.bind()

        # get sizeof float in bytes
        s = ctypes.sizeof(ctypes.c_float) * 3
        self.vbo.resize(num * 2 * s)
        offset = 0
        stride = s

        # packed attributes
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(
            0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(offset)
        )
        offset += num * s

        self.vao.unbind()
        self.program.stop()
        self._size = num

    def update(self, positions_1: torch.Tensor, positions_2: torch.Tensor):
        """
        Draw lines between positions_1 and positions_2. For each i, a line is drawn between positions_1[i] and positions_2[i].
        positions_1 and positions_2 are of shape (num_lines, 3).
        """
        num_lines = positions_1.shape[0]
        if num_lines == 0:
            return
        assert positions_1.shape[0] == positions_2.shape[0]
        assert isinstance(positions_1, torch.Tensor) and  isinstance(positions_2, torch.Tensor)
        self._resize(num_lines)

        temp = self.gl_buffer.map(dtype=wp.vec3f, shape=(2 * self._size,))
        temp_torch = wp.to_torch(temp).view(-1, 2, 3)
        temp_torch[:, 0, :] = positions_1
        temp_torch[:, 1, :] = positions_2
        self.gl_buffer.unmap()

    def draw(
        self,
        world2projT,
        line_width: float = 1.0,
        color:Tuple[float, float, float]=(0.078, 0.129, 0.239),
    ):
        self.program.use()
        self.vao.bind()
        gl.glLineWidth(line_width)
        self.program["world2proj"] = world2projT
        self.program["color"] = color
        gl.glDrawArrays(gl.GL_LINES, 0, self._size * 2)
        gl.glLineWidth(1.0)
        self.program.stop()
        self.vao.unbind()
