from typing import TYPE_CHECKING, Any, NoReturn, Sequence, Type
import warnings

import torch
import numpy as np

from pyglet.graphics import *
from pyglet.graphics.vertexdomain import _gl_types, _make_attribute_property, _nearest_pow2
from pyglet.graphics.shader import ShaderProgram
from pyglet.graphics import shader, vertexarray
import pyglet.gl as gl

from .cuda_buffer_object import AttributeCUDABufferObject

from pyglet.graphics.shader import Attribute
from pyglet.graphics.vertexarray import VertexArray
from pyglet.graphics.vertexbuffer import IndexedBufferObject

_debug_gl_shaders = pyglet.options['debug_gl_shaders']

class CUDAVertexDomain:
    """Management of a set of vertex lists.

    Construction of a vertex domain is usually done with the
    :py:func:`create_domain` function.
    """

    attribute_meta: dict[str, dict[str, Any]]
    buffer_attributes: list[tuple[AttributeCUDABufferObject, Attribute]]
    vao: VertexArray
    attribute_names: dict[str, Attribute]
    attrib_name_buffers: dict[str, AttributeCUDABufferObject]

    def __init__(self, attribute_meta: dict[str, dict[str, Any]]) -> None:  # noqa: D107
        self.attribute_meta = attribute_meta

        self.attribute_names = {}  # name: attribute
        self.buffer_attributes = []  # list of (buffer, attribute)
        self.attrib_name_buffers = {}  # dict of AttributeName: AttributeBufferObject (for VertexLists)

        self._property_dict = {}  # name: property(_getter, _setter)

        for name, meta in attribute_meta.items():
            assert meta['format'][0] in _gl_types, f"'{meta['format']}' is not a valid attribute format for '{name}'."
            location = meta['location']
            count = meta['count']
            gl_type = _gl_types[meta['format'][0]]
            normalize = 'n' in meta['format']
            instanced = meta['instance']

            self.attribute_names[name] = attribute = shader.Attribute(name, location, count, gl_type, normalize,
                                                                      instanced)

            # Create buffer:
            self.attrib_name_buffers[name] = buffer = AttributeCUDABufferObject(attribute)

            self.buffer_attributes.append((buffer, attribute))


        self.vao = vertexarray.VertexArray()
        self.vao.bind()
        for buffer, attribute in self.buffer_attributes:
            buffer.bind()
            attribute.enable()
            attribute.set_pointer(buffer.ptr)
            if attribute.instance:
                attribute.set_divisor()
        self.vao.unbind()
    
    def resize(self, num_elements: int) -> None:
        for buffer, _ in self.buffer_attributes:
            buffer.resize(num_elements)
    
    def get_buffer(self, name: str) -> AttributeCUDABufferObject:
        if name not in self.attrib_name_buffers:
            raise ValueError(f"Attribute '{name}' not found in the vertex domain. Available attributes: {list(self.attrib_name_buffers)}")
        return self.attrib_name_buffers[name]
    
    def update_buffer(self, name: str, data: torch.Tensor) -> None:
        buffer = self.get_buffer(name)
        buffer.update(data)

    def bind(self):
        self.vao.bind()
    
    def draw(self, mode: int, first: int, count: int):
        self.vao.bind()
        gl.glDrawArrays(mode, first, count)
        self.vao.unbind()
    
class IndexedCUDAVertexDomain(CUDAVertexDomain):
    def __init__(self, 
                 attribute_meta: dict[str, dict[str, Any]], 
                 indices: np.ndarray,
                 index_gl_type: int = gl.GL_UNSIGNED_INT) -> None:

        super().__init__(attribute_meta)
        self.index_gl_type = index_gl_type
        self.index_c_type = shader._c_types[index_gl_type]  # noqa: SLF001
        self.index_element_size = ctypes.sizeof(self.index_c_type)
        self.index_buffer = BufferObject(
            size=indices.nbytes,
            usage=gl.GL_DYNAMIC_DRAW)
        
        self.index_buffer.set_data(indices.tobytes())
        self.vao.bind()
        self.index_buffer.bind_to_index_buffer()
        self.vao.unbind()

def cuda_vertex_list_create(
        program: ShaderProgram, 
        indices: Sequence[int] | None = None,
        instances: Sequence[str] | None = None,
        **data: Any) -> CUDAVertexDomain:
    attributes = program._attributes.copy()

    indexed = indices is not None

    for name, fmt in data.items():
        try:
            attributes[name] = {**attributes[name], 'format': fmt, 'instance': name in instances if instances else False }
        except KeyError:  # noqa: PERF203
            if _debug_gl_shaders:
                msg = (f"The attribute `{name}` was not found in the Shader Program.\n"
                        f"Please check the spelling, or it may have been optimized out by the OpenGL driver.\n"
                        f"Valid names: {list(attributes)}")
                warnings.warn(msg)
            continue

    if _debug_gl_shaders:
        if missing_data := [key for key in attributes if key not in data]:
            msg = (
                f"No data was supplied for the following found attributes: `{missing_data}`.\n"
            )
            warnings.warn(msg)

    if indexed:
        domain = IndexedCUDAVertexDomain(attributes, indices)
    else:
        domain = CUDAVertexDomain(attributes)

    return domain