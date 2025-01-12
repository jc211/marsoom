import numpy as np
import torch

from pyglet import gl
from pyglet.graphics.shader import Shader, ShaderProgram
from .cuda_vertex_domain import cuda_vertex_list_create

ellipse_vertex_shader = """
#version 330 core


uniform WindowBlock
{
    mat4 projection;
    mat4 view;
} window;

uniform ViewportBlock {
    float width;
    float height;
} viewport;

layout (location = 0) in vec2 circle_pos_;

// Instance 
layout (location = 1) in vec3 pos_;
layout (location = 2) in vec3 color_;
layout (location = 3) in vec3 cov2d_; 
layout (location = 4) in float opacity_;

out vec4 color;

vec2 px2ndc(vec2 pos) {
    vec2 size = vec2(viewport.width, viewport.height);
    pos = pos/size;
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
    vec4 pos = window.projection * window.view * vec4(pos_, 1.0);
    pos /= pos.w;
    pos.xy = pos.xy + px2ndc(circle_pos);
    // bring to front
    gl_Position = pos;
    float alpha = opacity_;
    color = vec4(color_, alpha);
    //color = vec4(color_, 0.8);
}

"""

ellipse_fragment_shader = """
#version 330 core
in vec4 color;
out vec4 FragColor;
void main()
{
    FragColor = color;
}
"""


class EllipseRenderer:
    def __init__(self, num_segments: int = 20):

        self.program = ShaderProgram(
            Shader(ellipse_vertex_shader, "vertex"),
            Shader(ellipse_fragment_shader, "fragment"),
        )
        self.domain = cuda_vertex_list_create(
            program=self.program,
            instances=["pos_", "color_", "cov2d_", "opacity_"],
            circle_pos_="f",
            pos_="f",
            color_="f",
            cov2d_="f",
            opacity_="f"
        )
        self._define_circle(num_segments)
        self.num_segments = num_segments
        
    
    def _define_circle(self, num_segments: int):
        circle = np.zeros((num_segments, 2), dtype=np.float32)
        for i in range(num_segments):
            theta = 2.0 * np.pi * i / (num_segments - 1)
            circle[i, :] = [np.cos(theta), np.sin(theta)]
        circle = np.vstack((circle, np.zeros((1, 2), dtype=np.float32))) # add center
        self.domain.update_buffer("circle_pos_", torch.from_numpy(circle).cuda())

    def resize(self, num: int):
        self.domain.resize(num)
    
    def update_positions(self, positions: torch.Tensor):
        self.resize(positions.shape[0])
        self.domain.update_buffer("pos_", positions)
    
    def update_colors(self, colors: torch.Tensor):
        self.resize(colors.shape[0])
        self.domain.update_buffer("color_", colors)
    
    def update_cov2D(self, cov2D: torch.Tensor):
        self.resize(cov2D.shape[0])
        self.domain.update_buffer("cov2d_", cov2D)
    
    def update_opacity(self, opacity: torch.Tensor):
        self.resize(opacity.shape[0])
        self.domain.update_buffer("opacity_", opacity)
    
    def update(self, positions: torch.Tensor, colors: torch.Tensor, cov2D: torch.Tensor, opacity: torch.Tensor):
        self.resize(positions.shape[0])
        self.domain.update_buffer("pos_", positions)
        self.domain.update_buffer("color_", colors)
        self.domain.update_buffer("cov2d_", cov2D)
        self.domain.update_buffer("opacity_", opacity)


    def draw(
        self,
        line_width: float = 1.0,
    ):
        gl.glEnable(gl.GL_BLEND)
        gl.glLineWidth(line_width)
        self.program.use()
        self.domain.bind()
        gl.glDrawArraysInstanced(
            gl.GL_LINE_LOOP, 0, self.num_segments + 1, self.domain.num_elements
        )

