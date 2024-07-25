from pyglet import gl
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.graphics import vertexbuffer, vertexarray
from typing import Optional
import ctypes
import warp as wp
import numpy as np
import torch
from pydrake.trajectories import PiecewisePolynomial
from .point_renderer import PointRenderer

vertex_shader = """
#version 330 core
uniform mat4 world2proj;

layout (location = 0) in vec3 pos_;
layout (location = 1) in vec3 color_;
layout (location = 2) in float timestamp_;

out vec4 color;

void main() {
    gl_Position = world2proj * vec4(pos_, 1);
    //vec3 complementaryColor = vec3(1.0) - color_;
    //vec3 interpolatedColor = mix(color_, complementaryColor, timestamp_);
    // Shade color based on timestamp which is between 0 and 1
    color = vec4(color_, 0.3);
}
"""

fragment_shader = """
#version 330 core
//uniform vec3 color;

in vec4 color;

out vec4 FragColor;

void main()
{
    FragColor = vec4(color);
}

"""


class TrajectoryRenderer:
    def __init__(self, name:str = "noname"):
        self.name=name
        self.points = torch.tensor([], dtype=torch.float32) # shape(N, 3)
        self.colors = torch.tensor([], dtype=torch.float32) # shape(N, 3)
        self.timestamps = torch.tensor([], dtype=torch.float32) # shape(N, 1)
        self.line_segments = torch.tensor([], dtype=torch.int32) # shape(S, 2)
        self.point_renderer = PointRenderer()
        self.trajectory = None

        self.program = ShaderProgram(
            Shader(vertex_shader, "vertex"),
            Shader(fragment_shader, "fragment"),
        )
        self.vao = vertexarray.VertexArray()
        self.vbo = vertexbuffer.BufferObject(size=1)
        self.gl_buffer = wp.RegisteredGLBuffer(
            self.vbo.id, wp.get_cuda_device(), flags=wp.RegisteredGLBuffer.WRITE_DISCARD
        )
        self.ebo = vertexbuffer.BufferObject(size=1)
        self.ebo_gl_buffer = wp.RegisteredGLBuffer(
            self.ebo.id, wp.get_cuda_device(), flags=wp.RegisteredGLBuffer.WRITE_DISCARD
        )
        self._resize()

    def _resize(self, num_points: int = 1, num_segments: int = 1):
        from pyglet import gl

        self.program.use()
        self.vao.bind()
        self.vbo.bind()

        s_float = ctypes.sizeof(ctypes.c_float)
        point_dtype = 3*s_float + 3*s_float + 1*s_float
        
        self.vbo.resize(num_points * point_dtype)

        stride = point_dtype

        # POINT POSITIONS
        offset = 0
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(
            0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(offset)
        )
        offset += 3*s_float

        # POINT COLORS
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(
            1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(offset)
        )
        offset += 3*s_float

        # POINT TIMESTAMP
        gl.glEnableVertexAttribArray(2)
        gl.glVertexAttribPointer(
            2, 1, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(offset)
        )
        offset += s_float

        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo.id)
        self.ebo.resize(num_segments * 2 * ctypes.sizeof(ctypes.c_int32))

        self.vao.unbind()
        self.program.stop()
        self._size = num_points

    def update(self):
        num_points = self.points.shape[0]
        num_segments = self.line_segments.shape[0]
        assert num_points > 0, "No points to render"
        self._resize(num_points, num_segments)

        assert self.points.shape[0] == self.colors.shape[0] 
        assert self.points.shape[0] == self.timestamps.shape[0]

        temp = self.gl_buffer.map(dtype=wp.float32, shape=(self._size,7))
        temp_torch = wp.to_torch(temp)
        temp_torch[:, :3] = self.points
        temp_torch[:, 3:6] = self.colors
        temp_torch[:, 6] = self.timestamps.flatten()
        self.gl_buffer.unmap()

        temp = self.ebo_gl_buffer.map(dtype=wp.int32, shape=(self.line_segments.shape[0] * 2,))
        temp_torch = wp.to_torch(temp).view(-1, 2)
        temp_torch[:, 0] = self.line_segments[:, 0]
        temp_torch[:, 1] = self.line_segments[:, 1]
        self.ebo_gl_buffer.unmap()

    def create(self, 
               positions: list[list[float]], 
               color: list[float],
               timestamps: Optional[list[float]] = None,
               ):
        assert color is not None and len(color) == 3
        assert positions is not None and len(positions) > 1
        print(f"Creating trajectory with {len(positions)} points")

        if timestamps is None:
            timestamps = [0.0] * len(positions)

        line_segments = []
        good_positions = []
        good_timestamps = []



        self.min_timestamp = min(timestamps)
        self.max_timestamp = max(timestamps)

        reset_segment = True
        for i in range(len(positions)):
            t = timestamps[i]
            p = positions[i]
            if len(p) == 0:
                reset_segment = True
                continue
            good_positions.append(p)
            good_timestamps.append(t)
            idx = len(good_positions) - 1
            if not reset_segment:
                line_segments.append([idx - 1, idx])
            reset_segment = False

        assert len(good_positions) > 1
        assert len(line_segments) >0, "No line segments"

        points_np = np.array(good_positions, dtype=np.float32)
        line_segments_np = np.array(line_segments, dtype=np.int32)
        timestamps_np = np.array(good_timestamps, dtype=np.float32)
        print(f"Changed trajectory to have {points_np.shape[0]} points")

        self.trajectory = PiecewisePolynomial.FirstOrderHold(
            timestamps_np, points_np.T
        )

        self.timestamps = torch.tensor(timestamps_np, dtype=torch.float32).reshape(-1).cuda()
        # normalize
        self.timestamps = (self.timestamps - self.min_timestamp) / (self.max_timestamp - self.min_timestamp)
        self.points = torch.tensor(points_np, dtype=torch.float32).reshape(-1, 3).cuda()
        self.line_segments = torch.tensor(line_segments_np, dtype=torch.int32).reshape(-1, 2).cuda()
        self.colors = torch.tensor(color, dtype=torch.float32).reshape(1, 3).cuda().repeat(self.points.shape[0], 1).contiguous()
        self.update()

    def draw(
        self,
        world2projT,
        timestamp: Optional[float] = None,
        line_width: float = 1.0,
        point_size: float = 1.0,
    ):

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        self.program.use()
        self.vao.bind()
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo.id)
        gl.glLineWidth(line_width)
        gl.glPointSize(point_size)
        self.program["world2proj"] = world2projT
        # gl.glDrawElements(gl.GL_LINES, self.line_segments.shape[0] * 2, gl.GL_UNSIGNED_INT, None)
        gl.glDrawArrays(gl.GL_POINTS, 0, self.points.shape[0])
        gl.glLineWidth(1.0)
        gl.glPointSize(1.0)
        self.program.stop()
        self.vao.unbind()


        if timestamp is not None and self.trajectory is not None:
            pos = torch.tensor(self.trajectory.value(timestamp)).cuda().reshape(-1, 3)
            # color = torch.tensor([1.0, 0.0, 0.0, 1.0]).cuda().reshape(1, 4)
            color = self.colors[0].reshape(1, 3)
            color = torch.cat([color, torch.tensor([1.0]).cuda().reshape(1, 1)], dim=1)

            self.point_renderer.update(pos, color)
            self.point_renderer.draw(world2projT, point_size=20.0)

