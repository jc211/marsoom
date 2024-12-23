from pyglet.graphics.shader import ShaderProgram
from pyglet.graphics import Group
from pyglet import gl
from pyglet.math import Mat4
import pyglet


def get_default_shader() -> ShaderProgram:
    return pyglet.gl.current_context.create_program((LineGroup.default_vert_src, 'vertex'),
                                                    (LineGroup.default_frag_src, 'fragment'))

class LineGroup(Group):
    default_vert_src = """
    #version 330 core

    uniform WindowBlock
    {
        mat4 projection;
        mat4 view;
    } window;

    uniform mat4 model;

    in vec3 position;
    in vec4 color;
    out vec4 vert_color;

    void main() {
        gl_Position = window.projection * window.view * model * vec4(position, 1.0);
        vert_color = color;
    }
    """

    default_frag_src = """
    #version 330 core
    in vec4 vert_color;
    out vec4 FragColor;
    void main()
    {
        FragColor = vert_color; 
    }
    """

    def __init__(self, 
                 program: ShaderProgram, 
                 order: int = 0, 
                 parent: Group | None = None) -> None:
        super().__init__(order, parent)
        self.program = program
        self.matrix = Mat4()

    def set_state(self) -> None:
        self.program.use()
        self.program['model'] = self.matrix
    
    def __hash__(self) -> int:
        return hash((
            self.program, 
            self.order, 
            self.parent))

    def __eq__(self, other) -> bool:
        return (self.__class__ is other.__class__ and
                self.program == other.program and
                self.order == other.order and
                self.parent == other.parent and
                self.matrix == other.matrix
                )

class LineModel:
    def __init__(
        self,
        index,
        positions,
        colors,
        group: pyglet.graphics.Group = None,
        batch: pyglet.graphics.Batch = None,
    ):
        if batch is None:
            batch = pyglet.graphics.Batch()

        self.batch = batch
        self.group = group
        self.program = get_default_shader()
        mat_group = LineGroup(self.program, parent=self.group)
        self.position = positions
        self.colors = colors
        self.vlist = self.program.vertex_list_indexed(
            len(positions)//3, gl.GL_LINES, index, position=("f", positions), color=("f", colors),
            batch=self.batch, group=mat_group
        )
        self.groups = [mat_group]
        self._matrix = Mat4()
    
    def _update_vertices(self):
        self.vlist.position[:] = self.position
    
    def _update_colors(self):
        self.vlist.colors[:] = self.colors
    
    @property
    def matrix(self) -> Mat4:
        return self._matrix

    @matrix.setter
    def matrix(self, value: Mat4) -> None:
        self._matrix = value
        for group in self.groups:
            group.matrix = value

    def draw(self, line_width: float = 1.0):
        gl.glLineWidth(line_width)
        self.groups[0].set_state()  
        self.vlist.draw(gl.GL_LINES)
        self.groups[0].unset_state()
