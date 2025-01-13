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

    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec2 aTexCoord;

    out vec2 TexCoords;

    void main() {
        gl_Position = vec4(aPos, 1.0);
        TexCoords = aTexCoord;
    }
    """

    default_frag_src = """
    #version 330 core
    in vec2 TexCoords;

    out vec4 FragColor;

    uniform sampler2D textureSampler;
    uniform float alpha;

    void main() {
        vec4 color = texture(textureSampler, TexCoords);
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

    def set_state(self) -> None:
        self.program.use()
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

class Overlay:
    def __init__(self, 
                 tex_id: int,
                 alpha: float = 1.0,
                 group: Group | None = None,
                 batch: Batch | None = None) -> None:
        
        if batch is None:
            batch = Batch()

        self.batch = batch
        self.group = group

        verts = np.array(
            [
                -1.0, -1.0, 0.0,
                1.0, -1.0, 0.0,
                1.0, 1.0, 0.0,
                -1.0, 1.0, 0.0,
            ])
        
        tex = np.array(
            [
                0.0, 0.0,
                1.0, 0.0,
                1.0, 1.0,
                0.0, 1.0,
            ]
        )
        index = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        self.program = get_default_shader()
        mat_group = ImageQuadGroup(tex_id, self.program, parent=self.group)  
        self.vlist = self.program.vertex_list_indexed(
            4, gl.GL_TRIANGLES, 
            indices=index,
            aPos=("f", verts),
            aTexCoord=("f", tex),
            batch=self.batch, 
            group=mat_group
        )
        self.group = mat_group
        self.alpha = alpha
        self.tex_id = tex_id
    
    @property
    def tex_id(self) -> int:
        return self._tex_id 
    
    @tex_id.setter
    def tex_id(self, value: int) -> None:
        self._tex_id = value
        self.group.tex_id = value
    
    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        self._alpha = value
        self.group.alpha = value
    
    def draw(self) -> None:
        gl.glEnable(gl.GL_BLEND)
        self.group.set_state()
        self.batch.draw()   
