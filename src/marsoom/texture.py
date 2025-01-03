import ctypes

import numpy as np

# import torch
# import warp as wp
from pyglet import gl, image


class Texture:
    def __init__(self, 
                 width: int, 
                 height: int, 
                 fmt: int = gl.GL_RGB,
                 internal_format: int = gl.GL_RGBA
                 ):
        self.fmt = fmt
        self.internal_format = internal_format
        if fmt == gl.GL_RGB or fmt == gl.GL_BGR:
            self.dim = 3
        elif fmt == gl.GL_RGBA or fmt == gl.GL_BGRA:
            self.dim = 4
        elif fmt == gl.GL_DEPTH_COMPONENT:
            assert internal_format == gl.GL_DEPTH_COMPONENT32F, "internal format is inconsistent"
            self.dim =1
        else:
            raise NotImplementedError(f"{fmt} not implemented")
        


        # self.pbo = gl.GLuint()
        # gl.glGenBuffers(1, self.pbo)
        self.tex = None

        self.resize(width, height, dtype=gl.GL_UNSIGNED_BYTE)

        # self._pbo_to_texture()
        # self.cuda_pbo = wp.RegisteredGLBuffer(
        #     int(self.pbo.value),
        #     wp.get_cuda_device(),
        #     flags=wp.RegisteredGLBuffer.WRITE_DISCARD,
        # )
        # self.copy_from_device(torch.zeros((height, width, dim), dtype=torch.float32))
    
    def is_depth(self):
        return self.fmt == gl.GL_DEPTH_COMPONENT


    def __del__(self):
        try:
            gl.glDeleteBuffers(1, self.pbo)
        except:
            pass

    @property
    def id(self):
        return self.tex.id

    @property
    def width(self):
        return self.tex.width

    @property
    def height(self):
        return self.tex.height

    @property
    def aspect(self):
        return self.width / self.height

    def resize(self, width: int, height: int, dtype: int = gl.GL_FLOAT):
        if self.tex is not None:
            del self.tex
        self.tex = image.Texture.create(width=width, height=height, fmt=self.fmt, internalformat=self.internal_format)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex.id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_BORDER)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_BORDER)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        self.tex._set_tex_coords_order(3, 2, 1, 0)
        if dtype == gl.GL_FLOAT:
            self.element_size = ctypes.sizeof(ctypes.c_float)
        elif dtype == gl.GL_UNSIGNED_BYTE:
            self.element_size = ctypes.sizeof(ctypes.c_uint8)
        else:
            raise NotImplementedError(f"{dtype} not implemented")


        # gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self.pbo)
        # gl.glBufferData(
        #     gl.GL_PIXEL_UNPACK_BUFFER,
        #     width * height * self.dim * self.element_size,
        #     None,
        #     gl.GL_DYNAMIC_DRAW,
        # )
        # gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        self.dtype = dtype

    def copy_from_host(self, data: np.ndarray):

        dtype = gl.GL_FLOAT
        if data.dtype == np.float32:
            dtype = gl.GL_FLOAT
        elif data.dtype == np.uint8:
            dtype = gl.GL_UNSIGNED_BYTE
        else:
            raise NotImplementedError(f"{data.dtype} cannot be uploaded")

        if self.is_depth():
            assert data.ndim == 2 and data.dtype == np.float32
        else:
            assert data.ndim == 3 and data.shape[2] == self.dim

        w = data.shape[1]
        h = data.shape[0]
        if w != self.width or h != self.height or self.dtype != dtype:
            self.resize(w, h, dtype)

        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.id)
        gl.glTexSubImage2D(
            gl.GL_TEXTURE_2D,
            0,
            0,
            0,
            self.width,
            self.height,
            self.fmt,
            self.dtype,
            data.ctypes.data,
        )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    # def copy_from_device(self, data: torch.Tensor):
    #     raise NotImplementedType()
    # assert data.shape[2] == 3
    # w = data.shape[1]
    # h = data.shape[0]
    # if w != self.width or h != self.height:
    #     self.resize(w, h)
    # image = self.cuda_pbo.map(
    #     dtype=wp.float32, shape=(self.height * self.width * self.dim,)
    # )
    # image_torch = wp.to_torch(image)
    # image_torch.copy_(data.flatten())
    # self.cuda_pbo.unmap()
    # self._pbo_to_texture()

    def _pbo_to_texture(self):
        # Copy from pbo to texture
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.id)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self.pbo)
        gl.glTexParameteri(
            gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_BORDER
        )
        gl.glTexParameteri(
            gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_BORDER
        )
        # nearest interpolation
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexSubImage2D(
            gl.GL_TEXTURE_2D,
            0,
            0,
            0,
            self.width,
            self.height,
            self.fmt,
            self.dtype,
            ctypes.c_void_p(0),
        )
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def draw(self, width: int, height: int):
        self.tex.blit(0, 0, z=0, width=width, height=height)
