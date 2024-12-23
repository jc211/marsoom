import numpy as np
from pyglet import gl
from pyglet.graphics.shader import ShaderProgram
from pyglet.graphics import Group, Batch
from pyglet.math import Mat4

def get_default_shader() -> ShaderProgram:
    return gl.current_context.create_program((PointsGroup.default_vert_src, 'vertex'),
                                             (PointsGroup.default_frag_src, 'fragment'))

class PointsGroup(Group):
    default_vert_src = """
    #version 330 core

    uniform WindowBlock
    {
        mat4 projection;
        mat4 view;
    } window;

    uniform mat4 model;

    in vec3 position;
    in vec3 color;

    out vec4 color_;

    void main() {
        gl_Position = window.projection * window.view * model * vec4(position, 1.0);
        color_ = vec4(color, 1.0);
    }
    """

    default_frag_src = """
    #version 330 core

    out vec4 FragColor;

    in vec4 color_;

    void main()
    {
        FragColor = color_;

    }
    """

    def __init__(self, 
                 program: ShaderProgram, 
                 order: int = 0, 
                 parent: Group | None = None) -> None:
        super().__init__(order, parent)

        self.program = program
        self.matrix = Mat4()
        self.default_color = (1.0, 0.0, 0.0, 1.0)   

    def set_state(self) -> None:
        self.program.use()
        self.program['model'] = self.matrix
    
    def __hash__(self) -> int:
        return hash((self.program, self.order, self.parent))

    def __eq__(self, other) -> bool:
        return (self.__class__ is other.__class__ and
                self.program == other.program and
                self.order == other.order and
                self.parent == other.parent 
                )

class Points:
    def __init__(self, 
                 points: list[tuple[float, float, float]],
                 colors: list[tuple[float, float, float]],
                 group: Group | None = None,
                 batch: Batch | None = None) -> None:
        
        # colors should be between 0 and 1
        
        if batch is None:
            batch = Batch()
        
        self.points = points   
        self.colors = colors

        self.batch = batch
        self.group = group
        

        self.program = get_default_shader()
        mat_group = PointsGroup(self.program, parent=self.group)  
        num_points = len(points)
        self.vlist = self.program.vertex_list(
            num_points, gl.GL_POINTS, position=("f", self._get_vertices()), color=("f", self._get_colors()), batch=self.batch, group=mat_group
        )
        self._matrix = Mat4()   
        self.groups = [mat_group]
    
    def _get_vertices(self):
        return np.asarray(self.points, dtype=np.float32).flatten()
    
    def _get_colors(self):
        return np.asarray(self.colors, dtype=np.float32).flatten()
    
    def _update_vertices(self):
        self.vlist.position[:] = self._get_vertices()
    
    def _update_colors(self):
        self.vlist.color[:] = self._get_colors()
    
    def update(self, points: list[tuple[float, float, float]], colors: list[tuple[float, float, float]]) -> None:
        # colors should be between 0 and 1
        self.points = points
        self.colors = colors
        self._update_vertices()
        self._update_colors()

    
    @property
    def matrix(self) -> Mat4:
        return self._matrix

    @matrix.setter
    def matrix(self, value: Mat4) -> None:
        self._matrix = value
        for group in self.groups:
            group.matrix = value