import numpy as np
from pyglet import gl
from pyglet.graphics.shader import ShaderProgram
from pyglet.graphics import Group, Batch
from pyglet.math import Mat4

def get_default_shader() -> ShaderProgram:
    return gl.current_context.create_program((StructuredPointCloudGroup.default_vert_src, 'vertex'),
                                             (StructuredPointCloudGroup.default_frag_src, 'fragment'))

class StructuredPointCloudGroup(Group):
    default_vert_src = """
    #version 330 core

    uniform WindowBlock
    {
        mat4 projection;
        mat4 view;
    } window;

    uniform mat4 model;

    uniform sampler2D color_texture;

    uniform float fl_x;
    uniform float fl_y;
    uniform float cx;
    uniform float cy;
    uniform float width;
    uniform float height;


    in vec2 pixel;
    in float depth;
    out vec4 color_;

    void main() {
        float u = pixel.x / width;
        float v = (pixel.y / height);
        vec2 uv = vec2(u, v);

        float px = (pixel.x - cx) * depth / fl_x;
        float py = -(pixel.y - cy) * depth / fl_y;
        float pz = depth;
        vec3 position = vec3(px, py, -pz);

        gl_Position = window.projection * window.view * model * vec4(position, 1.0);
        color_ = texture(color_texture, uv);
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
                 color_texture_id: int,
                 width: int,
                 height: int,
                 fl_x: float = 1.0,
                 fl_y: float = 1.0,
                 cx: float = 0.0,
                 cy: float = 0.0,
                 order: int = 0, 
                 parent: Group | None = None) -> None:
        super().__init__(order, parent)

        self.program = program
        self.matrix = Mat4()
        self.color_texture_id = color_texture_id
        self.width = width
        self.height = height
        self.fl_x = fl_x
        self.fl_y = fl_y
        self.cx = cx
        self.cy = cy


    def set_state(self) -> None:
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.color_texture_id)
        self.program.use()
        self.program['model'] = self.matrix
        self.program['width'] = self.width
        self.program['height'] = self.height
        self.program['fl_x'] = self.fl_x
        self.program['fl_y'] = self.fl_y
        self.program['cx'] = self.cx
        self.program['cy'] = self.cy



    
    def __hash__(self) -> int:
        return hash((self.program, self.order, self.parent))

    def __eq__(self, other) -> bool:
        return (self.__class__ is other.__class__ and
                self.program == other.program and
                self.order == other.order and
                self.parent == other.parent 
                )

class StructuredPointCloud:
    def __init__(self, 
                 width: int,
                 height: int,
                 group: Group | None = None,
                 batch: Batch | None = None) -> None:
        
        # colors should be between 0 and 1
        
        if batch is None:
            batch = Batch()
        
        self._width = width
        self._height = height
        
        self.batch = batch
        self.group = group

        self._depth_texture_id = 0
        self._color_texture_id = 0
        self.fl_x = 1.0
        self.fl_y = 1.0
        self.cx = 0.0
        self.cy = 0.0

        self.program = get_default_shader()
        mat_group = StructuredPointCloudGroup(
            program=self.program, 
            fl_x=self.fl_x,
            fl_y=self.fl_y,
            cx=self.cx,
            cy=self.cy,
            color_texture_id=self._color_texture_id,
            width=self._width,
            height=self._height,
            parent=self.group)  
        num_points = self._width * self._height
        self.vlist = self.program.vertex_list(
            count=num_points, 
            mode=gl.GL_POINTS, 
            pixel=("f", self._get_vertices()), 
            depth=("f/stream", self._get_depth()),
            batch=self.batch, 
            group=mat_group
        )
        self._matrix = Mat4()   
        self.groups = [mat_group]
    
    def _get_vertices(self):
        # get pixels 
        x = np.arange(self.width, dtype=np.float32)
        y = np.arange(self.height, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        return np.stack([xx.flatten(), yy.flatten()], axis=1).flatten()
    
    def _get_depth(self):
        return np.zeros(self.width * self.height, dtype=np.float32)
    
    def _update_vertices(self):
        self.vlist.pixel[:] = self._get_vertices()
        self.vlist.depth[:] = self._get_depth()
    
    @property
    def width(self) -> int:
        return self._width
    
    @property
    def height(self) -> int:
        return self._height

    
    def update_intrinsics(self, fl_x: float, fl_y: float, cx: float, cy: float ) -> None:
        self.fl_x = fl_x
        self.fl_y = fl_y
        self.cx = cx
        self.cy = cy
        for group in self.groups:
            group.fl_x = fl_x
            group.fl_y = fl_y
            group.cx = cx
            group.cy = cy
    
    def update_depth(self, depth:np.ndarray) -> None:
        self.vlist.depth[:] = depth.flatten()
    

    @property
    def color_texture_id(self) -> int:
        return self._color_texture_id

    @color_texture_id.setter
    def color_texture_id(self, value: int) -> None:
        self._color_texture_id = value
        for group in self.groups:
            group.color_texture_id = value
    
    @property
    def matrix(self) -> Mat4:
        return self._matrix

    @matrix.setter
    def matrix(self, value: Mat4) -> None:
        self._matrix = value
        for group in self.groups:
            group.matrix = value
    