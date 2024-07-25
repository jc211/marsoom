import ctypes
import warp as wp
import torch

from pyglet import gl
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.graphics import vertexbuffer, vertexarray

pose_vertex_shader = """
#version 330 core
uniform mat4 world2proj;
uniform mat4 world;

layout (location = 0) in vec3 pos_;
layout (location = 1) in vec4 quat_;

out vec3 pos;
out vec4 quat;

void main() {
    pos = pos_;
    quat = quat_;
    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
}
"""

pose_geometry_shader = """
#version 330 core
layout (points) in;
layout (line_strip, max_vertices = 6) out;

uniform mat4 world2proj;
uniform float scale;

in vec3 pos[];
in vec4 quat[];

out vec4 color;

vec3 quat_rotate(vec4 q, vec3 v) {
    float w = q.w;
    vec3 xyz = q.xyz;
    vec3 t = 2.0 * cross(xyz, v);
    vec3 result = v + w * t + cross(xyz, t);
    return result;
}

void main() {
    mat4 x_vw = world2proj;
    vec4 quat = quat[0];

    //origin
    vec3 origin = pos[0];
    // x axis
    vec3 d_x = quat_rotate(quat, vec3(scale, 0.0, 0.0)) + origin;
    // y axis
    vec3 d_y = quat_rotate(quat, vec3(0.0, scale, 0.0)) + origin;
    // z axis
    vec3 d_z = quat_rotate(quat, vec3(0.0, 0.0, scale)) + origin;


    // Emit X Axis
    color = vec4(0.835,0.6186,0.45098,1.0);
    gl_Position = x_vw * vec4(origin, 1.0);
    EmitVertex();
    gl_Position = x_vw * vec4(d_x, 1.0);
    EmitVertex();

    // Emit Y Axis
    color = vec4(0.008, 0.6186, 0.45098, 1.0);
    gl_Position = x_vw * vec4(origin, 1.0);
    EmitVertex();
    gl_Position = x_vw * vec4(d_y, 1.0);
    EmitVertex();

    // Emit Z Axis
    color = vec4(0.0, 0.5843, 1.0, 1.0);
    gl_Position = x_vw * vec4(origin, 1.0);
    EmitVertex();
    gl_Position = x_vw * vec4(d_z, 1.0);
    EmitVertex();

    EndPrimitive();
}
"""

pose_fragment_shader = """
#version 330 core
in vec4 color;
out vec4 FragColor;
void main()
{
    FragColor = color; 
}
"""


class AxisRenderer:
    def __init__(self):

        self.program = ShaderProgram(
            Shader(pose_vertex_shader, "vertex"),
            Shader(pose_geometry_shader, "geometry"),
            Shader(pose_fragment_shader, "fragment"),
        )
        self.num_floats_per_element = [3, 4]  # position, quat
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

    def _resize(self, num: int):
        if num == self._num_points:
            return
        self._num_points = num
        struct_size = self.total_floats_per_element * ctypes.sizeof(ctypes.c_float)
        self.vbo.resize(num * struct_size)

    def update(self, positions: torch.Tensor, quats: torch.Tensor = None):
        """
        positions is of shape (num_points, 3)
        quats is of shape (num_points, 4) where quats are in xyzw format
        """
        assert isinstance(positions, torch.Tensor) and isinstance(quats, torch.Tensor)
        assert positions.shape[1] == 3
        assert quats.shape[1] == 4

        new_num_points = positions.shape[0]
        self._resize(new_num_points)

        temp = self.gl_buffer.map(
            dtype=wp.float32, shape=(self._num_points, self.total_floats_per_element)
        )
        temp_torch = wp.to_torch(temp, requires_grad=False)
        temp_torch[:, :3] = positions
        temp_torch[:, 3:7] = quats
        self.gl_buffer.unmap()

    def draw(
        self,
        world2projT,
        scale: float = 0.03,
        line_width: float = 1.0,
    ):
        self.program.use()
        self.vao.bind()
        gl.glLineWidth(line_width)
        self.program["world2proj"] = world2projT
        self.program["scale"] = scale
        gl.glDrawArrays(gl.GL_POINTS, 0, self._num_points)
        self.program.stop()
        self.vao.unbind()
