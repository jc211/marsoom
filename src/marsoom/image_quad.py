import numpy as np
from pyglet import gl
from pyglet.graphics.shader import ShaderProgram
from pyglet.graphics import Group, Batch
from pyglet.math import Mat4

def get_default_shader() -> ShaderProgram:
    return gl.current_context.create_program((ImageQuadGroup.default_vert_src, 'vertex'),
                                             (ImageQuadGroup.default_frag_src, 'fragment'))

class ImageQuadGroup(Group):
    default_vert_src = """
    #version 330 core

    uniform WindowBlock
    {
        mat4 projection;
        mat4 view;
    } window;

    uniform mat4 model;

    in vec3 position;
    in vec2 tex;
    out vec2 TexCoords;

    void main() {
        gl_Position = window.projection * window.view * model * vec4(position, 1.0);
        TexCoords = tex;
    }
    """

    default_frag_src = """
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

    def __init__(self, 
                 tex_id: int,
                 program: ShaderProgram, 
                 alpha: float = 1.0,
                 order: int = 0, 
                 parent: Group | None = None) -> None:
        super().__init__(order, parent)
        self.tex_id = tex_id    
        self.program = program
        self.alpha = alpha
        self.matrix = Mat4()

    def set_state(self) -> None:
        self.program.use()
        self.program['model'] = self.matrix
        self.program["alpha"] = self.alpha
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex_id)
    
    def __hash__(self) -> int:
        # return hash((self.program, self.order, self.parent, self.tex_id, self.alpha))
        return hash((self.program, self.order, self.parent))

    def __eq__(self, other) -> bool:
        return (self.__class__ is other.__class__ and
                self.program == other.program and
                self.order == other.order and
                self.parent == other.parent and
                # self.matrix == other.matrix and
                self.tex_id == other.tex_id and
                self.alpha == other.alpha
                )

class ImageQuad:
    def __init__(self, 
                 tex_id: int,
                 top_left: tuple[float, float, float],
                 top_right: tuple[float, float, float],
                 bot_right: tuple[float, float, float],
                 bot_left: tuple[float, float, float],
                 alpha: float = 1.0,
                 group: Group | None = None,
                 batch: Batch | None = None) -> None:
        
        if batch is None:
            batch = Batch()

        self.top_left = top_left
        self.top_right = top_right
        self.bot_right = bot_right
        self.bot_left = bot_left

        self.batch = batch
        self.group = group
        
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
        self.program = get_default_shader()
        mat_group = ImageQuadGroup(tex_id, self.program, parent=self.group)  
        self.vlist = self.program.vertex_list_indexed(
            4, gl.GL_TRIANGLES, index, position=("f", self._get_vertices()), tex=("f", tex), batch=self.batch, group=mat_group
        )
        self._matrix = Mat4()   
        self._alpha = alpha
        self._tex_id = tex_id
        self.groups = [mat_group]
    
    def _get_vertices(self):
        return np.array([
            *self.top_left,
            *self.top_right,
            *self.bot_right,
            *self.bot_left,
        ], dtype=np.float32)
    
    def _update_vertices(self):
        self.vlist.position[:] = self._get_vertices()
    
    def update(self, top_left: tuple[float, float, float], top_right: tuple[float, float, float], bot_right: tuple[float, float, float], bot_left: tuple[float, float, float]) -> None:
        if self.top_left == top_left and self.top_right == top_right and self.bot_right == bot_right and self.bot_left == bot_left:
            return
        self.top_left = top_left
        self.top_right = top_right
        self.bot_right = bot_right
        self.bot_left = bot_left
        self._update_vertices()
    
    @property
    def matrix(self) -> Mat4:
        return self._matrix

    @matrix.setter
    def matrix(self, value: Mat4) -> None:
        self._matrix = value
        for group in self.groups:
            group.matrix = value
    
    @property
    def tex_id(self) -> int:
        return self._tex_id 
    
    @tex_id.setter
    def tex_id(self, value: int) -> None:
        self._tex_id = value
        for group in self.groups:
            group.tex_id = value
    
    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        self._alpha = value
        for group in self.groups:
            group.alpha = value
