import ctypes

import numpy as np

import torch
import warp
from pyglet import gl, image



_numpy_to_gl = {
    np.dtype(np.float32): gl.GL_FLOAT,
    np.dtype(np.uint8): gl.GL_UNSIGNED_BYTE,
}
_gl_to_numpy = { v: k for k, v in _numpy_to_gl.items() }

_warp_to_gl = {
    warp.float32: gl.GL_FLOAT,
    warp.uint8: gl.GL_UNSIGNED_BYTE,
}


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

        self.cuda_available = warp.context.is_cuda_available()
        self.cuda_pbo = None
        self.tex = image.Texture.create(width=width, height=height, fmt=self.fmt, internalformat=self.internal_format)
        self.dtype = gl.GL_UNSIGNED_BYTE
        self.create_pbo(width, height, self.dtype)
        self.resize(width, height, dtype=self.dtype)
    
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
    
    def create_pbo(self, width, height, dtype):
        self.pbo = gl.GLuint()
        gl.glGenBuffers(1, self.pbo)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self.pbo)
        gl_data = np.zeros((height, width, self.dim), dtype=_gl_to_numpy[dtype])
        gl.glBufferData(gl.GL_PIXEL_UNPACK_BUFFER, gl_data.nbytes, gl_data.ctypes.data, gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        if self.cuda_available:
            self.cuda_pbo = warp.RegisteredGLBuffer(
                self.pbo,
                warp.get_cuda_device(),
                flags=warp.RegisteredGLBuffer.WRITE_DISCARD,
            )

    def resize(self, width: int, height: int, dtype: int = gl.GL_FLOAT):
        if self.width == width and self.height == height and self.dtype == dtype:
            return
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

        self.dtype = dtype

        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self.pbo)
        gl_data = np.zeros((height, width, self.dim), dtype=_gl_to_numpy[dtype])
        gl.glBufferData(gl.GL_PIXEL_UNPACK_BUFFER, gl_data.nbytes, gl_data.ctypes.data, gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)


    def copy_from_host(self, data: np.ndarray):
        assert data.dtype in _numpy_to_gl, f"{data.dtype} not supported - {list(_numpy_to_gl.keys())}"

        if self.is_depth():
            assert data.ndim == 2 and data.dtype == np.float32
        else:
            assert data.ndim == 3 and data.shape[2] == self.dim

        h, w = data.shape[:2]
        self.resize(w, h, dtype=_numpy_to_gl[data.dtype])

        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self.pbo)
        gl.glBufferData(gl.GL_PIXEL_UNPACK_BUFFER, data.nbytes, data.ctypes.data, gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        self._pbo_to_texture()

        # gl.glActiveTexture(gl.GL_TEXTURE0)
        # gl.glBindTexture(gl.GL_TEXTURE_2D, self.id)
        # gl.glTexSubImage2D(
        #     gl.GL_TEXTURE_2D,
        #     0,
        #     0,
        #     0,
        #     self.width,
        #     self.height,
        #     self.fmt,
        #     self.dtype,
        #     data.ctypes.data,
        # )
        # gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def copy_from_device(self, data: torch.Tensor):
        data = warp.from_torch(data)
        assert self.cuda_available, "CUDA not available"
        assert data.dtype in _warp_to_gl, f"{data.dtype} not supported"

        if self.is_depth():
            assert data.ndim == 2 and data.dtype == np.float32
        else:
            assert data.ndim == 3 and data.shape[2] == self.dim

        h, w = data.shape[:2]
        self.resize(w, h, _warp_to_gl[data.dtype])
        image = self.cuda_pbo.map(
            dtype=data.dtype, 
            shape=(self.height * self.width * self.dim,)
        )
        warp.copy(image, data)
        self.cuda_pbo.unmap()
        self._pbo_to_texture()

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
