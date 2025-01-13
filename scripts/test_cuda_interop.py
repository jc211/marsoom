import warp as wp
import numpy as np
from pyglet.gl import *

wp.init()

# create a GL buffer
gl_buffer_id = GLuint()
glGenBuffers(1, gl_buffer_id)

# copy some data to the GL buffer
glBindBuffer(GL_ARRAY_BUFFER, gl_buffer_id)
gl_data = np.arange(1024, dtype=np.float32)
glBufferData(GL_ARRAY_BUFFER, gl_data.nbytes, gl_data.ctypes.data, GL_DYNAMIC_DRAW)
glBindBuffer(GL_ARRAY_BUFFER, 0)

# register the GL buffer with CUDA
cuda_gl_buffer = wp.RegisteredGLBuffer(gl_buffer_id)

# map the GL buffer to a Warp array
arr = cuda_gl_buffer.map(dtype=wp.float32, shape=(1024,))
# launch a Warp kernel to manipulate or read the array
# unmap the GL buffer
cuda_gl_buffer.unmap()