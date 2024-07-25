import ctypes
import warp as wp
import numpy as np
import torch

from pyglet import gl
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.graphics import vertexbuffer, vertexarray

sphere_vertex_renderer = """
#version 330 core

uniform float image_width;
uniform float image_height;
uniform mat4 world2proj;
uniform mat4 model;

layout (location = 0) in vec2 circle_pos_;

// Instance 
layout (location = 1) in vec3 pos_;
layout (location = 2) in float radius_;

out vec4 color;

vec2 px2ndc(vec2 pos) {
    vec2 viewport = vec2(image_width, image_height);
    pos = pos/viewport;
    pos = pos*2.0;
    pos.y = -pos.y;
    return pos;
}

vec2 scale_by_cov(mat2 cov2d, vec2 p) {
    // https://github.com/Kitware/VTK/blob/acb5d75143a6d28203525549e7ca040c6de21584/Rendering/OpenGL2/glsl/vtkPointGaussianVS.glsl#L118

    mat2 transformVCVSOutput;
    if (abs(cov2d[0][1]) > 1e-6)
    {
        float halfTrace = 0.5 * (cov2d[0][0] + cov2d[1][1]);
        float term = sqrt(halfTrace * halfTrace - determinant(cov2d));
        float eigenValue1 = halfTrace + term;
        float eigenValue2 = halfTrace - term;

        vec2 eigenVector1 = normalize(vec2(eigenValue1 - cov2d[1][1], cov2d[0][1]));
        vec2 eigenVector2 = normalize(vec2(eigenVector1.y, -eigenVector1.x));
        transformVCVSOutput = mat2(sqrt(eigenValue1) * eigenVector1, sqrt(eigenValue2) * eigenVector2);
    }
    else
    {
        transformVCVSOutput = mat2(sqrt(cov2d[0][0]), 0.0, 0.0, sqrt(cov2d[1][1]));
    }

    return transformVCVSOutput*p;

}

void main() {

    mat2 cov2d = mat2(
        cov2d_[0], cov2d_[1],
        cov2d_[1], cov2d_[2] 
    );
    
    vec2 circle_pos = scale_by_cov(cov2d, circle_pos_);
    vec4 pos = world2proj * model * vec4(pos_, 1.0);
    pos /= pos.w;
    pos.xy = pos.xy + px2ndc(circle_pos);
    // bring to front
    gl_Position = pos;
    color = vec4(color_, 1.0);
}

"""

sphere_fragment_shader = """
#version 330 core
in vec4 color;
out vec4 FragColor;
void main()
{
    FragColor = color;
}
"""


class SphereRenderer:
    def __init__(self, num_segments: int = 20):

        self.program = ShaderProgram(
            Shader(sphere_vertex_renderer, "vertex"),
            Shader(sphere_fragment_shader, "fragment"),
        )
        self.num_floats_per_element = [3, 1]  # position, radius
        self.total_floats_per_element = sum(self.num_floats_per_element)
        self.num_segments = num_segments
        self._generate_gl_objects()

    def _generate_gl_objects(self):

        self.vao = vertexarray.VertexArray()
        self.circle_vbo = vertexbuffer.BufferObject(
            size=self.num_segments * 2 * ctypes.sizeof(ctypes.c_float)
        )
        self.vbo = vertexbuffer.BufferObject(size=1)
        self.gl_buffer = wp.RegisteredGLBuffer(
            self.vbo.id, wp.get_cuda_device(), flags=wp.RegisteredGLBuffer.WRITE_DISCARD
        )

        self.program.use()
        self.vao.bind()

        gl.glEnableVertexAttribArray(0)
        self.circle_vbo.bind()
        num_segments = self.num_segments
        circle = np.zeros((num_segments, 2), dtype=np.float32)
        for i in range(num_segments):
            theta = 2.0 * np.pi * i / (num_segments - 1)
            circle[i, :] = [np.cos(theta), np.sin(theta)]
        # add center point
        circle = np.vstack((circle, np.zeros((1, 2), dtype=np.float32)))
        self.circle_vbo.set_data(circle.tobytes())
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, ctypes.c_void_p(0))
        self.circle_vbo.unbind()

        self.vbo.bind()
        offset = 0
        stride = self.total_floats_per_element * ctypes.sizeof(ctypes.c_float)
        for i, num_floats in enumerate(self.num_floats_per_element):
            gl.glEnableVertexAttribArray(i + 1)
            gl.glVertexAttribPointer(
                i + 1,
                num_floats,
                gl.GL_FLOAT,
                gl.GL_FALSE,
                stride,
                ctypes.c_void_p(offset),
            )
            offset += num_floats * ctypes.sizeof(ctypes.c_float)
            gl.glVertexAttribDivisor(i + 1, 1)
        self.vbo.unbind()

        self.vao.unbind()
        self.program.stop()
        self._num_points = 0

    def _resize(self, num: int):
        self._num_points = num
        struct_size = self.total_floats_per_element * ctypes.sizeof(ctypes.c_float)
        self.vbo.resize(num * struct_size)

    def _copy(self, positions: torch.Tensor, radius: torch.Tensor):
        temp = self.gl_buffer.map(
            dtype=wp.float32, shape=(self._num_points, self.total_floats_per_element)
        )
        temp_torch = wp.to_torch(temp)
        with torch.no_grad():
            start = 0
            for num_floats, tensor in zip(
                self.num_floats_per_element, [positions, radius]
            ):
                temp_torch[:, start : (start + num_floats)] = tensor
                start += num_floats
        self.gl_buffer.unmap()

    def update(self, positions: torch.Tensor, radius: torch.Tensor):
        """Positions in pixels, cov2D in pixels"""
        assert positions.shape[1] == 3
        assert radius.shape[1] == 1

        if positions.device != torch.device("cuda"):
            positions = positions.cuda()
        new_num_points = positions.shape[0]
        if new_num_points != self._num_points:
            self._resize(new_num_points)
        self._copy(positions=positions, radius=radius)

    def draw(
        self,
        world2projT: np.ndarray,
        image_width: float,
        image_height: float,
        model=np.eye(4, dtype=np.float32),
        line_width: float = 1.0,
    ):
        gl.glLineWidth(line_width)
        self.program.use()
        self.vao.bind()
        self.program["image_width"] = image_width
        self.program["image_height"] = image_height
        self.program["world2proj"] = world2projT
        self.program["model"] = model.flatten()
        gl.glDrawArraysInstanced(
            gl.GL_LINE_LOOP, 0, self.num_segments + 1, self._num_points
        )

        self.program.stop()
        self.vao.unbind()
