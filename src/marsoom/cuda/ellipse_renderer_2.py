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
layout (location = 2) in vec4 quat_;
layout (location = 3) in vec3 scale_;
layout (location = 4) in vec3 color_;
layout (location = 5) in float opacity_;

out vec4 color;

vec2 px2ndc(vec2 pos) {
    vec2 size = vec2(viewport.width, viewport.height);
    pos = pos/size;
    pos = pos*2.0;
    pos.y = -pos.y;
    return pos;
}

mat3 quat_to_rotmat(vec4 quat) {
    float w = quat.x, x = quat.y, y = quat.z, z = quat.w; 
    // normalize
    float inv_norm = inversesqrt(x * x + y * y + z * z + w * w);
    x *= inv_norm;
    y *= inv_norm;
    z *= inv_norm;
    w *= inv_norm;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y, wz = w * z;
    return mat3(
        (1.0 - 2.0 * (y2 + z2)),
        (2.0 * (xy + wz)),
        (2.0 * (xz - wy)), // 1st col
        (2.0 * (xy - wz)),
        (1.0 - 2.0 * (x2 + z2)),
        (2.0 * (yz + wx)), // 2nd col
        (2.0 * (xz + wy)),
        (2.0 * (yz - wx)),
        (1.0 - 2.0 * (x2 + y2)) // 3rd col
    );
}

mat3 quat_scale_to_covariance(
    mat3 R, 
    vec3 scale 
) {
    mat3 S = mat3(scale.x, 0.0, 0.0, 
                  0.0, scale.y, 0.0, 
                  0.0, 0.0, scale.z);
    mat3 M = R * S;
    mat3 covar = M * transpose(M);
    return covar;
}

mat3 projection_jacobian(vec4 p) {
    float flx = window.projection[0][0]*window.projection[3][2];
    float fly = window.projection[1][1]*window.projection[3][2];
    float tx = window.view[3][0];
    float ty = window.view[3][1];
    float z = p.z/p.w;  
    return mat3(
        flx/z, 0.0, 0.0,
        0.0, fly/z, 0.0,
        -flx*tx/(z*z), -fly*ty/(z*z), 0.0
    );
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
    vec4 pos_camera = window.view * vec4(pos_, 1.0);
    vec4 pos = window.projection * pos_camera;
    mat3 R = quat_to_rotmat(quat_);

    mat3 J = projection_jacobian(pos_camera);
    mat3 cov3d = quat_scale_to_covariance(R, scale_);
    mat3 W = mat3(window.view);

    mat3 cov2d_ = J * W * cov3d * transpose(W) * transpose(J);
    mat2 cov2d = mat2(cov2d_);
    vec2 circle_pos = scale_by_cov(cov2d, circle_pos_);
    pos /= pos.w;
    pos.xy = pos.xy + circle_pos;
    // bring to front
    gl_Position = pos;
    float alpha = opacity_;
    color = vec4(color_, alpha);
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
            instances=["pos_", "quat_", "scale_", "color_", "opacity_"],
            circle_pos_="f",
            pos_="f",
            quat_="f",  
            scale_="f",
            color_="f",
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
    
    def update_opacity(self, opacity: torch.Tensor):
        self.resize(opacity.shape[0])
        self.domain.update_buffer("opacity_", opacity)
    
    def update_quats(self, quats: torch.Tensor):
        # SCALAR FIRST
        self.resize(quats.shape[0])
        self.domain.update_buffer("quat_", quats)
    
    def update_scales(self, scales: torch.Tensor):
        self.resize(scales.shape[0])
        self.domain.update_buffer("scale_", scales)
    
    def update(self, 
               positions: torch.Tensor, 
               colors: torch.Tensor,  
               quats: torch.Tensor,  # SCALAR FIRST
               scales: torch.Tensor, 
               opacity: torch.Tensor):
        self.resize(positions.shape[0])
        self.domain.update_buffer("pos_", positions)
        self.domain.update_buffer("color_", colors)
        self.domain.update_buffer("quat_", quats)
        self.domain.update_buffer("scale_", scales)
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

