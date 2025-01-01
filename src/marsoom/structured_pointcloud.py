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

    uniform sampler2D depth_texture;
    uniform sampler2D color_texture;

    uniform float fl_x;
    uniform float fl_y;
    uniform float cx;
    uniform float cy;
    uniform float width;
    uniform float height;
    uniform float depth_scale;


    in vec2 pixel;
    out vec4 color_;

    void main() {
        float u = pixel.x / width;
        float v = 1.0 - pixel.y / height;
        vec2 uv = vec2(u, v);
        float depth = texture(depth_texture, uv).r * depth_scale;

        float px = (pixel.x - cx) * depth / fl_x;
        float py = -(pixel.y - cy) * depth / fl_y;
        float pz = -depth;
        vec3 position = vec3(px, py, pz);

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
                 depth_texture_id: int,
                 color_texture_id: int,
                 depth_scale: float,
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
        self.depth_texture_id = depth_texture_id 
        self.color_texture_id = color_texture_id
        self.depth_scale = depth_scale
        self.width = width
        self.height = height
        self.fl_x = fl_x
        self.fl_y = fl_y
        self.cx = cx
        self.cy = cy


    def set_state(self) -> None:
        self.program.use()
        self.program['model'] = self.matrix
        self.program['depth_scale'] = self.depth_scale
        self.program['width'] = self.width
        self.program['height'] = self.height
        self.program['fl_x'] = self.fl_x
        self.program['fl_y'] = self.fl_y
        self.program['cx'] = self.cx
        self.program['cy'] = self.cy

        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.depth_texture_id)
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.color_texture_id)

    
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
        self._depth_scale = 1.0

        self.program = get_default_shader()
        mat_group = StructuredPointCloudGroup(
            program=self.program, 
            fl_x=self.fl_x,
            fl_y=self.fl_y,
            cx=self.cx,
            cy=self.cy,
            depth_texture_id=self._depth_texture_id,
            color_texture_id=self._color_texture_id,
            width=self._width,
            height=self._height,
            depth_scale=self._depth_scale,
            parent=self.group)  
        num_points = self._width * self._height
        self.vlist = self.program.vertex_list(
            num_points, gl.GL_POINTS, position=("f", self._get_vertices()), batch=self.batch, group=mat_group
        )
        self._matrix = Mat4()   
        self.groups = [mat_group]
    
    def _get_vertices(self):
        # get pixels 
        x = np.arange(self.width)
        y = np.arange(self.height)
        xx, yy = np.meshgrid(x, y)
        return np.stack([xx.flatten(), yy.flatten()], axis=1)
    
    def _update_vertices(self):
        self.vlist.position[:] = self._get_vertices()
    
    
    @property
    def width(self) -> int:
        return self._width
    
    @width.setter
    def width(self, value: int) -> None:
        self._width = value
        self._update_vertices()
        for group in self.groups:
            group.width = value
    
    @property
    def height(self) -> int:
        return self._height

    @height.setter
    def height(self, value: int) -> None:
        self._height = value
        self._update_vertices()
        for group in self.groups:
            group.height = value
    
    def update_intrinsics(self, fl_x: float, fl_y: float, cx: float, cy: float, depth_scale: float) -> None:
        self.fl_x = fl_x
        self.fl_y = fl_y
        self.cx = cx
        self.cy = cy
        self._depth_scale = depth_scale
        for group in self.groups:
            group.fl_x = fl_x
            group.fl_y = fl_y
            group.cx = cx
            group.cy = cy
            group.depth_scale = depth_scale
    
    @property
    def depth_texture_id(self) -> int:
        return self._depth_texture_id
    
    @depth_texture_id.setter
    def depth_texture_id(self, value: int) -> None:
        self._depth_texture_id = value
        for group in self.groups:
            group.depth_texture_id = value

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