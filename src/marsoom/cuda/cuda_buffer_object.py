from typing import TYPE_CHECKING, Any, Sequence
from contextlib import contextmanager
import torch
import warp as wp
import ctypes

import pyglet.gl as gl
from pyglet.graphics import vertexbuffer
from pyglet.graphics.shader import Attribute


ctype_to_torch_dtype_dict = {
    ctypes.c_float: torch.float32,
    ctypes.c_double: torch.float64,
    ctypes.c_bool: torch.bool,
    ctypes.c_int16: torch.int16,
    ctypes.c_int32: torch.int32,
    ctypes.c_int64: torch.int64,
    ctypes.c_uint8: torch.uint8,
    ctypes.c_uint16: torch.uint16,
    ctypes.c_uint32: torch.uint32,
    ctypes.c_uint64: torch.uint64,
    ctypes.c_long: torch.int64,
    ctypes.c_byte: torch.int8,
    ctypes.c_char: torch.int8,
    ctypes.c_short: torch.int16,
    ctypes.c_int: torch.int32,
}
torch_dtype_to_warp_dtype = {
    torch.float32: wp.float32,
    torch.float64: wp.float64,
    torch.bool: wp.bool,
    torch.int16: wp.int16,
    torch.int32: wp.int32,
    torch.int64: wp.int64,
    torch.uint8: wp.uint8,
    torch.uint16: wp.uint16,
    torch.uint32: wp.uint32,
    torch.uint64: wp.uint64,
    torch.int8: wp.int8,
}

class CUDABackedVertexBuffer(vertexbuffer.BufferObject):
    def __init__(self, 
                 ctype,
                 count: int,
                 usage: int = gl.GL_DYNAMIC_DRAW, 
                 flags: int = wp.RegisteredGLBuffer.WRITE_DISCARD
                 ):

        if not ctype in ctype_to_torch_dtype_dict:
            raise ValueError(f"Unsupported ctype: {ctype}")

        self.ctype = ctype
        self.dtype = ctype_to_torch_dtype_dict[ctype]
        self.shape = torch.Size((1, count))
        self.count = count
        size = self.shape.numel() * self.dtype.itemsize
        super().__init__(size, usage)
        self._gl_buffer = wp.RegisteredGLBuffer(
            int(self.id),
            wp.get_cuda_device(),
            flags
        )
    
    def resize(self, num_elements: int):
        if num_elements == self.shape[0]:
            return
        self.shape = torch.Size((num_elements, self.count))
        size = self.shape.numel() * self.dtype.itemsize
        super().resize(size)
    
    def update(self, data: torch.Tensor):
        assert data.dtype == self.dtype, f"Data dtype {data.dtype} does not match buffer dtype {self.dtype}"
        assert data.shape[1] == self.count, f"Data shape {data.shape} does not match count {self.count}"
        if data.shape[0] != self.shape[0]:
            self.resize(data.shape[0])
        
        with self.tensor() as temp:
            temp.copy_(data)

    @contextmanager
    def tensor(self):
        temp = self._gl_buffer.map(
            dtype=torch_dtype_to_warp_dtype[self.dtype],
            shape=self.shape,
        )
        temp_torch = wp.to_torch(temp, requires_grad=False)
        yield temp_torch
        self._gl_buffer.unmap()
    

class AttributeCUDABufferObject(CUDABackedVertexBuffer):
    """A backed buffer used for Shader Program attributes."""

    def __init__(self, attribute: Attribute) -> None:  # noqa: D107
        super().__init__(attribute.c_type, attribute.count)
    

