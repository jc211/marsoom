import numpy as np
from pyglet import gl
from pyglet.graphics.shader import ShaderProgram
from pyglet.graphics import Group, Batch
from pyglet.graphics.vertexbuffer import BufferObject
from pyglet.graphics.vertexarray import VertexArray
from pyglet.math import Mat4
from .vertex_domain import vertex_list_create
import ctypes

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
    uniform float depth_scale;


    layout(location = 0) in vec2 pixel;
    layout(location = 1) in float depth;
    out vec4 color_;

    void main() {
        if (depth == 0.0 || depth == 1.0) {
            gl_Position = vec4(0.0, 0.0, 0.0, -1.0);
            return;
        }
        float u = pixel.x / width;
        float v = (pixel.y / height);
        vec2 uv = vec2(u, v);
        float depth_f = depth*depth_scale;

        float px = (pixel.x - cx) * depth_f / fl_x;
        float py = -(pixel.y - cy) * depth_f / fl_y;
        float pz = depth_f;
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
                 depth_scale: float = 1.0,
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
        self.depth_scale = depth_scale


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
        self.program['depth_scale'] = self.depth_scale



    
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
                 depth_scale: float = 1.0,
                 ):
                 
        
        # colors should be between 0 and 1
        
        
        self._width = width
        self._height = height

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
            height=self._height)
        self.num_points = self._width * self._height

        self.vao = VertexArray()
        self.pixel_buffer = BufferObject(
            size=ctypes.sizeof(ctypes.c_float) * 2 * self.num_points,
            usage=gl.GL_STATIC_DRAW
            )
        self.depth_buffer = BufferObject(
            size=ctypes.sizeof(ctypes.c_uint16) * self.num_points,
            usage=gl.GL_STREAM_DRAW
        )
        with self.vao:
            gl.glActiveTexture(gl.GL_TEXTURE0)

            self.pixel_buffer.bind()
            gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, 0)
            gl.glEnableVertexAttribArray(0)
            self.pixel_buffer.unbind()  

            self.depth_buffer.bind()
            gl.glEnableVertexAttribArray(1)
            # gl.glVertexAttribIPointer(1, 1, gl.GL_UNSIGNED_SHORT, 0, 0)
            gl.glVertexAttribPointer(1, 1, gl.GL_UNSIGNED_SHORT, gl.GL_TRUE, 0, 0)
            self.depth_buffer.unbind()

        self.update_vertices(self._get_vertices())
        self.update_depth(self._get_depth())
        self._matrix = Mat4()   
        self.group = mat_group
        self.depth_scale = depth_scale
    
    def update_vertices(self, vertices: np.ndarray) -> None:
        assert vertices.dtype == np.float32
        if self.pixel_buffer.size != vertices.nbytes:
            self.pixel_buffer.resize(vertices.nbytes)
        self.pixel_buffer.set_data(vertices.ctypes.data)
    
    def update_depth(self, depth: np.ndarray) -> None:
        assert depth.dtype == np.uint16
        if self.depth_buffer.size != depth.nbytes:
            self.depth_buffer.resize(depth.nbytes)
        self.depth_buffer.set_data(depth.ctypes.data)
        
    
    def _get_vertices(self):
        # get pixels 
        x = np.arange(self.width, dtype=np.float32)
        y = np.arange(self.height, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        return np.stack([xx.flatten(), yy.flatten()], axis=1).flatten()
    
    def _get_depth(self):
        return np.zeros(self.width * self.height, dtype=np.uint16)
    
    
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

        self.group.fl_x = fl_x
        self.group.fl_y = fl_y
        self.group.cx = cx
        self.group.cy = cy
    
    @property
    def depth_scale(self) -> float:
        return self.group.depth_scale

    @depth_scale.setter
    def depth_scale(self, value: float) -> None:
        self.group.depth_scale = value*65535.0

    @property
    def color_texture_id(self) -> int:
        return self._color_texture_id

    @color_texture_id.setter
    def color_texture_id(self, value: int) -> None:
        self._color_texture_id = value
        self.group.color_texture_id = value
    
    @property
    def matrix(self) -> Mat4:
        return self._matrix

    @matrix.setter
    def matrix(self, value: Mat4) -> None:
        self._matrix = value
        self.group.matrix = value
    
    def draw(self):
        self.group.set_state()
        self.vao.bind()
        gl.glDrawArrays(gl.GL_POINTS, 0, self.num_points)
    