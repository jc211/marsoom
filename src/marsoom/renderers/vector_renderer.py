import ctypes
import warp as wp
import torch

from pyglet import gl
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.graphics import vertexbuffer, vertexarray

from marsoom.context_3d import Context3D

vertex_shader = """
#version 330 core
uniform mat4 world2proj;
uniform mat4 world;

layout (location = 0) in vec3 pos_;
layout (location = 1) in vec3 dir_;

out vec3 pos;
out vec3 dir;

void main() {
    pos = pos_;
    dir = dir_;
    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
}
"""

geometry_shader = """
#version 330 core
layout (points) in;
layout (line_strip, max_vertices = 2) out;

uniform mat4 world2proj;
uniform float vector_scale;

in vec3 pos[];
in vec3 dir[];

out vec4 color;

void main() {
    vec3 direction = dir[0];
    float tmin = length(direction)*vector_scale;
    direction = normalize(direction);

    mat4 x_vw = world2proj;

    gl_Position = x_vw * vec4(pos[0], 1.0); 
    color = vec4(1.0, 0.8, 0.0, 1.0);
    EmitVertex();
    gl_Position = x_vw * vec4(pos[0] + tmin*direction, 1.0);
    color = vec4(1.0, 0.8, 0.0, 1.0);
    EmitVertex();
    EndPrimitive();
}
"""

fragment_shader = """
#version 330 core
in vec4 color;
out vec4 FragColor;
void main()
{
    FragColor = color; 
}
"""


class VectorRenderer:
    def __init__(self):
        self.program = ShaderProgram(
            Shader(vertex_shader, "vertex"),
            Shader(geometry_shader, "geometry"),
            Shader(fragment_shader, "fragment"),
        )
        self.vao = vertexarray.VertexArray()
        self.vbo = vertexbuffer.BufferObject(size=1)
        self.gl_buffer = wp.RegisteredGLBuffer(
            self.vbo.id, wp.get_cuda_device(), flags=wp.RegisteredGLBuffer.WRITE_DISCARD
        )
        self._resize()

    def _resize(self, num: int = 1):
        from pyglet import gl

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

        # directions
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(
            1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(offset)
        )
        offset += num * s

        self.vao.unbind()
        self.program.stop()
        self._size = num

    def _copy(self, pos: wp.array(dtype=wp.vec3f), dir: wp.array(dtype=wp.vec3f)):
        num = pos.shape[0]
        temp = self.gl_buffer.map(dtype=wp.vec3f, shape=(2 * self._size,))
        offset = int(0)
        wp.copy(dest=temp, src=pos, dest_offset=offset, count=num)  # type: ignore
        offset += num
        wp.copy(dest=temp, src=dir, dest_offset=offset, count=num)  # type: ignore
        self.gl_buffer.unmap()

    def update(self, positions: torch.Tensor, directions: torch.Tensor):
        if isinstance(positions, torch.Tensor):
            positions = wp.from_torch(positions, dtype=wp.vec3f, requires_grad=False)
        if isinstance(directions, torch.Tensor):
            directions = wp.from_torch(directions, dtype=wp.vec3f, requires_grad=False)

        new_size = positions.shape[0]
        if new_size != self._size:
            self._resize(new_size)
        self._copy(positions, directions)

    def draw(
        self,
        context: Context3D,
        vector_scale: float = 0.03,
        line_width: float = 1.0,
        start_idx: int = 0
    ):
        self.program.use()
        self.vao.bind()
        gl.glLineWidth(line_width)
        self.program["world2proj"] = context.world2projT
        self.program["vector_scale"] = vector_scale
        gl.glDrawArrays(gl.GL_POINTS, start_idx, self._size)
        gl.glLineWidth(1.0)
        self.program.stop()
        self.vao.unbind()
