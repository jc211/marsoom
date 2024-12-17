from __future__ import annotations

from typing import TYPE_CHECKING

import pyglet
from pyglet import graphics
from pyglet.math import Mat4, Vec4
import trimesh
from pathlib import Path

if TYPE_CHECKING:
    from pyglet.graphics import Group
    from pyglet.graphics.shader import ShaderProgram


def get_default_shader() -> ShaderProgram:
    return pyglet.gl.current_context.create_program((MaterialGroup.default_vert_src, 'vertex'),
                                                    (MaterialGroup.default_frag_src, 'fragment'))



class BaseMaterialGroup(graphics.Group):
    default_vert_src: str
    default_frag_src: str
    matrix: Mat4 = Mat4()
    color: Vec4 = Vec4(1.0, 0.0, 0.0, 1.0)

    def __init__(self, program: ShaderProgram, order: int = 0, parent: Group | None = None) -> None:
        super().__init__(order, parent)
        self.program = program


class MaterialGroup(BaseMaterialGroup):
    default_vert_src = """#version 330 core
    in vec3 position;
    in vec3 normals;
    in vec4 colors;

    out vec3 vertex_normals;
    out vec3 vertex_position;

    uniform WindowBlock
    {
        mat4 projection;
        mat4 view;
    } window;

    uniform mat4 model;

    void main()
    {
        vec4 pos = window.view * model * vec4(position, 1.0);
        gl_Position = window.projection * pos;
        mat3 normal_matrix = transpose(inverse(mat3(model)));

        vertex_position = pos.xyz;
        vertex_normals = normal_matrix * normals;
    }
    """
    default_frag_src = """#version 330 core
    in vec3 vertex_normals;
    in vec3 vertex_position;
    out vec4 final_colors;

    uniform vec4 color;

    void main()
    {
        float ambientStrength = 0.3;
        vec3 lightColor = vec3(1.0, 1.0, 1.0);
        vec3 ambient = ambientStrength * lightColor;

        vec3 sun_direction = normalize(vec3(1.0, 1.0, 1.0));
        float diff = max(dot(normalize(vertex_normals), sun_direction), 0.0);
        vec3 diffuse = diff * lightColor;

        vec3 result = (ambient + diffuse) * color.rgb;

        final_colors = vec4(result, color.a);
    }
    """

    def set_state(self) -> None:
        self.program.use()
        self.program['model'] = self.matrix
        self.program['color'] = self.color
    
    def __hash__(self) -> int:
        return hash((self.program, self.order, self.parent))

    def __eq__(self, other) -> bool:
        return False
        return (self.__class__ is other.__class__ and
                self.program == other.program and
                self.order == other.order and
                self.parent == other.parent)
    

class ColoredModel(pyglet.model.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._color = Vec4(1.0, 0.0, 0.0, 1.0)
    
    @property
    def color(self) -> Vec4:
        return self._color

    @color.setter
    def color(self, value: Vec4) -> None:
        self._color = value
        for group in self.groups:
            if isinstance(group, MaterialGroup):
                group.color = value


class STLModelDecoder(pyglet.model.codecs.ModelDecoder):
    def get_file_extensions(self):
        return ['.stl']

    def decode(self, filename, file, batch, group=None):
            if not batch:
                batch = pyglet.graphics.Batch()

            m = trimesh.load_mesh(filename)
            faces = []
            vertices = []
            normals = []

            for v in m.vertices:
                vertices.append(float(v[0]))
                vertices.append(float(v[1]))
                vertices.append(float(v[2]))

            for f in m.faces:
                faces.append(int(f[0]))
                faces.append(int(f[1]))
                faces.append(int(f[2]))
            
            vertex_normals = trimesh.geometry.mean_vertex_normals(len(m.vertices), m.faces, m.face_normals)
            
            for n in vertex_normals:
                normals.append(float(n[0]))
                normals.append(float(n[1]))
                normals.append(float(n[2]))


            vertex_lists = []
            groups = []

            program = get_default_shader()
            matgroup = MaterialGroup(program, parent=group)
            vertex_lists.append(program.vertex_list_indexed(len(vertices)//3, pyglet.gl.GL_TRIANGLES, batch=batch, group=matgroup,
                                                    indices=faces,
                                                    position=('f', vertices),
                                                    normals=('f', normals),
                                                    ))
            groups.append(matgroup)

            return ColoredModel(vertex_lists=vertex_lists, groups=groups, batch=batch)

def get_decoders():
    return [STLModelDecoder()]


def get_encoders():
    return []
