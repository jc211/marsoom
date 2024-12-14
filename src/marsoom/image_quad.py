import numpy as np

from pyglet import gl
from pyglet.graphics.shader import Shader, ShaderProgram

from marsoom.context_3d import Context3D

quad_vertex_shader = """
#version 330 core
uniform mat4 world2proj;
uniform mat4 world;
in vec3 position;
in vec2 tex;
out vec2 TexCoords;
void main() {
    gl_Position = world2proj * world * vec4(position, 1.0);
    TexCoords = tex;
}
"""

quad_fragment_shader = """
#version 330 core
uniform float alpha;
out vec4 FragColor;
in vec2 TexCoords;
uniform sampler2D texture1;
void main()
{
    //FragColor = texture(texture1, TexCoords);
    vec4 color = texture(texture1, TexCoords);
    FragColor = vec4(color.rgb, alpha);

}
"""


class ImageQuad:
    def __init__(self, positions):

        tex = np.array(
            [
                0.0,
                0.0,
                1.0,
                0.0,
                1.0,
                1.0,
                0.0,
                1.0,
            ]
        )
        index = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
        self.program = ShaderProgram(
            Shader(quad_vertex_shader, "vertex"),
            Shader(quad_fragment_shader, "fragment"),
        )
        self.vlist = self.program.vertex_list_indexed(
            4, gl.GL_TRIANGLES, index, position=("f", positions), tex=("f", tex)
        )
        self.x_wv = np.eye(4, dtype=np.float32)

    def update_position(self, x_wv):
        self.x_wv = x_wv

    def draw(self, context: Context3D, tex_id, alpha=1.0):

        self.program.use()
        self.program["world2proj"] = context.world2projT
        self.program["world"] = self.x_wv.T.flatten()
        self.program["alpha"] = alpha
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
        self.vlist.draw(mode=gl.GL_TRIANGLES)
        self.program.stop()
