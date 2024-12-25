import ctypes

import numpy as np

# import torch
# import warp as wp
from pyglet import gl, image


class Texture:
    def __init__(self, width: int, height: int, dim: int = 3):
        assert dim == 3 or dim == 4
        self.dim = dim
        self.element_size = ctypes.sizeof(ctypes.c_float)
        self.tex = image.Texture.create(width=width, height=height, fmt=gl.GL_RGB)
        self.tex._set_tex_coords_order(3, 2, 1, 0)
        self.pbo = gl.GLuint()
        gl.glGenBuffers(1, self.pbo)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self.pbo)
        gl.glBufferData(
            gl.GL_PIXEL_UNPACK_BUFFER,
            width * height * self.dim * self.element_size,
            None,
            gl.GL_DYNAMIC_DRAW,
        )
        self._pbo_to_texture()
        # self.cuda_pbo = wp.RegisteredGLBuffer(
        #     int(self.pbo.value),
        #     wp.get_cuda_device(),
        #     flags=wp.RegisteredGLBuffer.WRITE_DISCARD,
        # )
        # self.copy_from_device(torch.zeros((height, width, dim), dtype=torch.float32))

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

    def resize(self, width: int, height: int):
        del self.tex
        self.tex = image.Texture.create(width=width, height=height)
        self.tex._set_tex_coords_order(3, 2, 1, 0)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self.pbo)
        gl.glBufferData(
            gl.GL_PIXEL_UNPACK_BUFFER,
            width * height * self.dim * self.element_size,
            None,
            gl.GL_DYNAMIC_DRAW,
        )
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)

    def copy_from_host(self, data: np.ndarray):
        assert data.shape[2] == 3
        assert data.dtype == np.float32
        w = data.shape[1]
        h = data.shape[0]
        if w != self.width or h != self.height:
            self.resize(w, h)

        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.id)
        gl.glTexSubImage2D(
            gl.GL_TEXTURE_2D,
            0,
            0,
            0,
            self.width,
            self.height,
            gl.GL_RGB,
            gl.GL_FLOAT,
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
            gl.GL_RGB,
            gl.GL_FLOAT,
            ctypes.c_void_p(0),
        )
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def draw(self, width: int, height: int):
        self.tex.blit(0, 0, z=0, width=width, height=height)
