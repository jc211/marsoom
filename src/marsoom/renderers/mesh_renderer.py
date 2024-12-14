import numpy as np
import torch
import warp as wp
import ctypes
from pathlib import Path

from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet import gl
from pyglet.graphics import vertexbuffer, vertexarray

from marsoom.context_3d import Context3D

shape_vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

// column vectors of the instance transform matrix
layout (location = 3) in vec3 position;
layout (location = 4) in vec4 rotation;
layout (location = 5) in vec3 scaling;
layout (location = 6) in vec3 color;

uniform mat4 world2proj;
uniform mat4 model;
uniform float scale_modifier;

out vec3 Normal;
out vec3 FragPos;
out vec2 TexCoord;
out vec3 ObjectColor1;
out vec3 ObjectColor2;

mat4 X(vec4 q, vec3 p, vec3 s) {
    q = normalize(q);
    float x = q.y;
    float y = q.z;
    float z = q.w;
    float w = q.x;
    return mat4(
        s.x * (1.0 - 2.0 * y * y - 2.0 * z * z), s.x * (2.0 * x * y + 2.0 * w * z), s.x * (2.0 * x * z - 2.0 * w * y), 0.0,
        s.y * (2.0 * x * y - 2.0 * w * z), s.y * (1.0 - 2.0 * x * x - 2.0 * z * z), s.y * (2.0 * y * z + 2.0 * w * x), 0.0,
        s.z * (2.0 * x * z + 2.0 * w * y), s.z * (2.0 * y * z - 2.0 * w * x), s.z * (1.0 - 2.0 * x * x - 2.0 * y * y), 0.0,
        p.x, p.y, p.z, 1.0
    );
};


void main()
{
    mat4 transform = model * X(rotation, position, scale_modifier*scaling);
    vec4 worldPos = transform * vec4(aPos, 1.0);
    gl_Position = world2proj * worldPos;
    //gl_Position = vec4(aPos, 1.0);
    FragPos = vec3(worldPos);
    Normal = mat3(transpose(inverse(transform))) * aNormal;

    TexCoord = aTexCoord;
    ObjectColor1 = color;
    ObjectColor2 = color;
}
"""

shape_fragment_shader = """
#version 330 core
out vec4 FragColor;

in vec3 Normal;
in vec3 FragPos;
in vec2 TexCoord;
in vec3 ObjectColor1;
in vec3 ObjectColor2;

uniform vec3 viewPos;
uniform vec3 lightColor;
uniform vec3 sunDirection;
uniform float alpha;

void main()
{
    float ambientStrength = 0.3;
    vec3 ambient = ambientStrength * lightColor;
    vec3 norm = normalize(Normal);

    float diff = max(dot(norm, sunDirection), 0.0);
    vec3 diffuse = diff * lightColor;

    vec3 lightDir2 = normalize(vec3(1.0, 0.3, -0.3));
    diff = max(dot(norm, lightDir2), 0.0);
    diffuse += diff * lightColor * 0.3;

    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - FragPos);

    vec3 reflectDir = reflect(-sunDirection, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;

    reflectDir = reflect(-lightDir2, norm);
    spec = pow(max(dot(viewDir, reflectDir), 0.0), 64);
    specular += specularStrength * spec * lightColor * 0.3;

    // checkerboard pattern
    float u = TexCoord.x;
    float v = TexCoord.y;
    // blend the checkerboard pattern dependent on the gradient of the texture coordinates
    // to void Moire patterns
    vec2 grad = abs(dFdx(TexCoord)) + abs(dFdy(TexCoord));
    float blendRange = 1.5;
    float blendFactor = max(grad.x, grad.y) * blendRange;
    float scale = 2.0;
    float checker = mod(floor(u * scale) + floor(v * scale), 2.0);
    checker = mix(checker, 0.5, smoothstep(0.0, 1.0, blendFactor));

    vec3 checkerColor = mix(ObjectColor1, ObjectColor2, checker);

    //vec3 result = (ambient + diffuse + specular) * checkerColor;
    vec3 result = (ambient + diffuse + 0.00001*specular) * checkerColor;
    FragColor = vec4(result, alpha);
}
"""


class MeshRenderer:

    @staticmethod
    def from_file(filename: str | Path, scale: float = 1.0):
        filename = Path(filename)
        assert filename.exists() and filename.is_file()
        import trimesh
        faces = []
        vertices = []
        m = trimesh.load(filename)
        for v in m.vertices:
            vertices.append(np.array(v) * scale)

        for f in m.faces:
            faces.append(int(f[0]))
            faces.append(int(f[1]))
            faces.append(int(f[2]))
        return MeshRenderer(np.array(vertices, dtype=np.float32), np.asarray(faces, np.uint32))


    def __init__(self, vertices: np.ndarray, indices:np.ndarray):
        assert vertices.dtype == np.float32
        assert indices.dtype == np.uint32
        self.program = ShaderProgram(
            Shader(shape_vertex_shader, "vertex"),
            Shader(shape_fragment_shader, "fragment"),
        )
        self.vertices = vertices
        self.indices = indices
        self.num_floats_per_element = [3, 4, 3, 3]  # position, rotation, scaling, color
        self.total_floats_per_element = sum(self.num_floats_per_element)
        self._generate_gl_objects()
        self.num_instances = 0

    def _generate_mesh_object(self):
        vertices = self.vertices
        indices = self.indices

        self.mesh_vbo = vertexbuffer.BufferObject(size=1, usage=gl.GL_STATIC_DRAW)
        self.mesh_vbo.bind()
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices.ctypes.data, gl.GL_STATIC_DRAW
        )
        # Set up vertex attributes
        vertex_stride = vertices.shape[1] * vertices.itemsize
        # positions
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(
            0, 3, gl.GL_FLOAT, gl.GL_FALSE, vertex_stride, ctypes.c_void_p(0)
        )
        # normals
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(
            1,
            3,
            gl.GL_FLOAT,
            gl.GL_FALSE,
            vertex_stride,
            ctypes.c_void_p(3 * vertices.itemsize),
        )
        # uv coordinates
        gl.glEnableVertexAttribArray(2)
        gl.glVertexAttribPointer(
            2,
            2,
            gl.GL_FLOAT,
            gl.GL_FALSE,
            vertex_stride,
            ctypes.c_void_p(6 * vertices.itemsize),
        )

        # self.mesh_vbo.set_data(vertices.tobytes())
        self.mesh_ebo = vertexbuffer.BufferObject(size=1, usage=gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.mesh_ebo.id)
        gl.glBufferData(
            gl.GL_ELEMENT_ARRAY_BUFFER,
            indices.nbytes,
            indices.ctypes.data,
            gl.GL_STATIC_DRAW,
        )

        self.face_count = len(indices)

    def _generate_gl_objects(self):
        self.vao = vertexarray.VertexArray()
        self.vbo = vertexbuffer.BufferObject(size=1)
        self.gl_buffer = wp.RegisteredGLBuffer(
            self.vbo.id, wp.get_cuda_device(), flags=wp.RegisteredGLBuffer.WRITE_DISCARD
        )

        self.program.use()
        self.vao.bind()
        self._generate_mesh_object()
        self.vbo.bind()

        offset = 0
        stride = self.total_floats_per_element * ctypes.sizeof(ctypes.c_float)
        for i, num_floats in enumerate(self.num_floats_per_element):
            gl.glEnableVertexAttribArray(i + 3)
            gl.glVertexAttribPointer(
                i + 3,
                num_floats,
                gl.GL_FLOAT,
                gl.GL_FALSE,
                stride,
                ctypes.c_void_p(offset),
            )
            offset += num_floats * ctypes.sizeof(ctypes.c_float)
            gl.glVertexAttribDivisor(i + 3, 1)

        self.vao.unbind()
        self.program.stop()

    def _resize(self, num: int):
        self.vbo.resize(
            num * self.total_floats_per_element * ctypes.sizeof(ctypes.c_float)
        )
        self.num_instances = num

    def update(
        self,
        positions: torch.Tensor,
        rotations: torch.Tensor,
        scaling: torch.Tensor,
        colors: torch.Tensor,
    ):
        assert positions.shape[1] == 3
        assert rotations.shape[1] == 4
        assert scaling.shape[1] == 3
        assert colors.shape[1] == 3
        new_num_instances = positions.shape[0]
        if new_num_instances != self.num_instances:
            self._resize(new_num_instances)
        temp = self.gl_buffer.map(
            dtype=wp.float32, shape=(self.num_instances, self.total_floats_per_element)
        )
        temp_torch = wp.to_torch(temp)
        with torch.no_grad():
            start = 0
            for num_floats, tensor in zip(
                self.num_floats_per_element, [positions, rotations, scaling, colors]
            ):
                temp_torch[:, start : (start + num_floats)] = tensor
                start += num_floats
        self.gl_buffer.unmap()

    def draw(
        self,
        context: Context3D,
        scale_modifier: float = 1.0,
        model=np.eye(4, dtype=np.float32),
        sun_direction=np.array([0.2, -0.8, 0.3], dtype=np.float32),
        alpha=1.0,
    ):
        sun_direction = np.asarray(sun_direction, dtype=np.float32)
        sun_direction /= np.linalg.norm(sun_direction)
        self.program.use()
        self.vao.bind()
        self.program["world2proj"] = context.world2projT
        self.program["model"] = model.flatten()
        self.program["lightColor"] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.program["sunDirection"] = sun_direction
        self.program["viewPos"] = context.camera_positon
        self.program["scale_modifier"] = scale_modifier
        self.program["alpha"] = alpha
        gl.glDrawElementsInstanced(
            gl.GL_TRIANGLES,
            self.face_count,
            gl.GL_UNSIGNED_INT,
            None,
            self.num_instances,
        )
        self.program.stop()


def create_sphere_mesh(
    radius=1.0, num_latitudes=32, num_longitudes=32, reverse_winding=False
):
    vertices = []
    indices = []

    for i in range(num_latitudes + 1):
        theta = i * np.pi / num_latitudes
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        for j in range(num_longitudes + 1):
            phi = j * 2 * np.pi / num_longitudes
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)

            x = cos_phi * sin_theta
            y = cos_theta
            z = sin_phi * sin_theta

            u = float(j) / num_longitudes
            v = float(i) / num_latitudes

            vertices.append([x * radius, y * radius, z * radius, x, y, z, u, v])

    for i in range(num_latitudes):
        for j in range(num_longitudes):
            first = i * (num_longitudes + 1) + j
            second = first + num_longitudes + 1

            if reverse_winding:
                indices.extend(
                    [first, second, first + 1, second, second + 1, first + 1]
                )
            else:
                indices.extend(
                    [first, first + 1, second, second, first + 1, second + 1]
                )

    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)
