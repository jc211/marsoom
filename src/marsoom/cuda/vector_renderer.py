import ctypes
import warp as wp
import torch

from pyglet import gl
from pyglet.graphics.shader import ShaderProgram
from pyglet.graphics import vertexbuffer, vertexarray

from .cuda_vertex_domain import cuda_vertex_list_create

def get_default_shader() -> ShaderProgram:
    return gl.current_context.create_program((vertex_shader, 'vertex'),
                                              (geometry_shader, 'geometry'),
                                             (fragment_shader, 'fragment')) 

vertex_shader = """
#version 330 core

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

uniform WindowBlock
{
    mat4 projection;
    mat4 view;
} window;

uniform float vector_scale;

in vec3 pos[];
in vec3 dir[];

out vec4 color;

void main() {
    vec3 direction = dir[0];
    float tmin = length(direction)*vector_scale;
    direction = normalize(direction);

    mat4 world2proj = window.projection * window.view;
    gl_Position = world2proj * vec4(pos[0], 1.0); 
    color = vec4(1.0, 0.8, 0.0, 1.0);
    EmitVertex();
    gl_Position = world2proj * vec4(pos[0] + tmin*direction, 1.0);
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
        self.program = get_default_shader()
        self.domain = cuda_vertex_list_create(
            program=self.program,
            pos_="f",
            dir_="f",
        )
    
    def update(self, positions: torch.Tensor, directions: torch.Tensor):
        self.domain.resize(positions.shape[0])
        self.domain.update_buffer("pos_", positions)
        self.domain.update_buffer("dir_", directions)

    def draw(
        self,
        vector_scale: float = 0.03,
        line_width: float = 1.0,
        start_idx: int = 0
    ):
        self.program.use()
        self.domain.bind()
        gl.glLineWidth(line_width)
        self.program["vector_scale"] = vector_scale
        gl.glDrawArrays(gl.GL_POINTS, start_idx, self._size)
        gl.glLineWidth(1.0)
