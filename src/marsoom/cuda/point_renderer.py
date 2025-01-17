import torch
from pyglet import gl
from pyglet.graphics import Group, Batch
from pyglet.graphics.shader import Shader, ShaderProgram


from .cuda_vertex_domain import cuda_vertex_list_create

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

    layout (location = 0) in vec3 position;
    layout (location = 1) in vec4 color;

    out vec4 color_;

    void main() {
        color_ = color;
        gl_Position = window.projection * window.view * vec4(position, 1.0);
    }

    """

    default_frag_src = """
    #version 330 core
    in vec4 color_;
    out vec4 FragColor;
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

    def set_state(self) -> None:
        self.program.use()
    

class PointRenderer:
    def __init__(self):
        self.group = PointsGroup(get_default_shader())
        self.domain = cuda_vertex_list_create(
            program=self.group.program,
            position="f",
            color="f",
        )

    def update_positions_and_colors(self, positions: torch.Tensor, colors: torch.Tensor):
        assert positions.shape[0] == colors.shape[0]
        self.domain.update_buffer("position", positions)
        self.domain.update_buffer("color", colors)
    
    def num_points(self):
        return self.domain.get_buffer("position").shape[0]

    def draw(
        self,
        point_size: float = 10.0,
        start_index: int = 0,
    ):
        gl.glPointSize(point_size)
        self.group.set_state()
        self.domain.draw(gl.GL_POINTS, start_index, self.num_points())
