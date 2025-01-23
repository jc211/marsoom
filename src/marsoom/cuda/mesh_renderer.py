from typing import Optional
from dataclasses import dataclass

import numpy as np
import torch

from pyglet.graphics.shader import ShaderProgram
from pyglet import gl
from pyglet.graphics import Group

from .cuda_vertex_domain import cuda_vertex_list_create


def get_default_shader() -> ShaderProgram:
    return gl.current_context.create_program((MeshGroup.default_vertex_shader, 'vertex'),
                                             (MeshGroup.default_fragment_shader, 'fragment'))


class MeshGroup(Group):
    toon_vert_src = """
    #version 330 core

    uniform WindowBlock
    {
        mat4 projection;
        mat4 view;
    } window;

    uniform float scale_modifier;

    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aNormal;

    // column vectors of the instance transform matrix
    layout (location = 2) in vec3 position;
    layout (location = 3) in vec4 rotation;
    layout (location = 4) in vec3 scaling;
    layout (location = 5) in vec3 color;

    mat4 X(vec4 q, vec3 p, vec3 s) {
        q = normalize(q);
        float w = q.x;
        float x = q.y;
        float y = q.z;
        float z = q.w;
        return mat4(
            s.x * (1.0 - 2.0 * y * y - 2.0 * z * z), s.x * (2.0 * x * y + 2.0 * w * z), s.x * (2.0 * x * z - 2.0 * w * y), 0.0,
            s.y * (2.0 * x * y - 2.0 * w * z), s.y * (1.0 - 2.0 * x * x - 2.0 * z * z), s.y * (2.0 * y * z + 2.0 * w * x), 0.0,
            s.z * (2.0 * x * z + 2.0 * w * y), s.z * (2.0 * y * z - 2.0 * w * x), s.z * (1.0 - 2.0 * x * x - 2.0 * y * y), 0.0,
            p.x, p.y, p.z, 1.0
        );
    };

    out vec3 BaseColor;
    out vec3 Normal;

    void main()
    {
        mat4 transform = X(rotation, position, scale_modifier*scaling);
        vec4 worldPos = transform * vec4(aPos, 1.0);
        gl_Position = window.projection * window.view * worldPos;
        //gl_Position = vec4(aPos, 1.0);
        Normal = mat3(transpose(inverse(transform))) * aNormal;
        BaseColor = color;
    }
    """

    toon_frag_src = """
        #version 330 core
        out vec4 FragColor;

        in vec3 BaseColor;
        in vec3 Normal;

        uniform vec3 sunDirection;

        void main()
        {
            // Calculate grayscale based on baseColor
            float intensity = dot(sunDirection, Normal); 

            // Toon shading thresholds (adjust as needed)
            float threshold1 = 0.2; 
            float threshold2 = 0.5;
            float threshold3 = 0.8;

            // Determine the toon shade based on grayscale and baseColor
            if (intensity < threshold1) {
                FragColor = vec4(BaseColor * 0.1, 1.0); // Darkest shade
            } else if (intensity < threshold2) {
                FragColor = vec4(BaseColor * 0.4, 1.0); // Mid-tone
            } else if (intensity < threshold3) {
                FragColor = vec4(BaseColor * 0.7, 1.0); // Lighter shade
            } else {
                FragColor = vec4(BaseColor, 1.0); // Brightest shade
            }

        }
        """

    default_vertex_shader = """
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aNormal;

    // column vectors of the instance transform matrix
    layout (location = 2) in vec3 position;
    layout (location = 3) in vec4 rotation;
    layout (location = 4) in vec3 scaling;
    layout (location = 5) in vec3 color;

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



    uniform WindowBlock
    {
        mat4 projection;
        mat4 view;
    } window;

    uniform float scale_modifier;

    out vec3 Normal;
    out vec3 FragPos;
    out vec3 BaseColor;

    void main()
    {
        mat4 transform = X(rotation, position, scale_modifier*scaling);
        vec4 worldPos = transform * vec4(aPos, 1.0);
        gl_Position = window.projection * window.view * worldPos;
        Normal = mat3(transpose(inverse(transform))) * aNormal;
        FragPos = vec3(worldPos);
        BaseColor = color;
    }
    """

    default_fragment_shader = """
    #version 330 core
    out vec4 FragColor;

    in vec3 Normal;
    in vec3 FragPos;
    in vec3 BaseColor;
    

    uniform LightBlock {
        vec3 viewPos;
        vec3 lightColor;
        vec3 sunDirection;
    } light;

    void main()
    {
        float ambientStrength = 0.3;
        vec3 ambient = ambientStrength * light.lightColor;
        vec3 norm = normalize(Normal);

        float diff = max(dot(norm, light.sunDirection), 0.0);
        vec3 diffuse = diff * light.lightColor;

        vec3 lightDir2 = normalize(vec3(1.0, 0.3, -0.3));
        diff = max(dot(norm, lightDir2), 0.0);
        diffuse += diff * light.lightColor * 0.3;

        float specularStrength = 0.5;
        vec3 viewDir = normalize(light.viewPos - FragPos);

        vec3 reflectDir = reflect(-light.sunDirection, norm);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
        vec3 specular = specularStrength * spec * light.lightColor;

        reflectDir = reflect(-lightDir2, norm);
        spec = pow(max(dot(viewDir, reflectDir), 0.0), 64);
        specular += specularStrength * spec * light.lightColor * 0.3;


        vec3 result = (ambient + diffuse + specular) * BaseColor;
        FragColor = vec4(result, 1.0);
    }
    """

    def __init__(self, 
                 program: ShaderProgram, 
                 sun_direction: np.ndarray,
                 scale_modifier: float = 1.0,
                 order: int = 0, 
                 parent: Group | None = None) -> None:
        super().__init__(order, parent)

        self.program = program
        self.sun_direction = sun_direction
        self.scale_modifier = scale_modifier

    def set_state(self) -> None:
        self.program.use()
        self.program["scale_modifier"] = self.scale_modifier



class InstancedMeshRenderer:

    # @staticmethod
    # def from_open3d_mesh(mesh: o3d.geometry.TriangleMesh, default_color: np.ndarray = np.array([1.0, 0.0, 0.0])):
    #     mesh.compute_vertex_normals()   
    #     return InstancedMeshRenderer(np.array(mesh.vertices), np.array(mesh.triangles), np.array(mesh.vertex_normals), default_color)

    def __init__(self, 
                 vertices: np.ndarray,
                 triangles: np.ndarray,
                 normals: np.ndarray,
                 default_color: np.ndarray = np.array([1.0, 0.0, 0.0])):

        self.group = MeshGroup(get_default_shader(), sun_direction=np.array([0.2, -0.8, 0.3], dtype=np.float32))
        indices = np.array(triangles).flatten().astype(np.uint32)
        self.index_count = len(indices)
        self.domain = cuda_vertex_list_create(
            program=self.group.program,
            indices=indices,
            instances=["position", "rotation", "scaling", "color"],
            position="f",
            color="f",
            scaling="f",
            rotation="f",
            aNormal="f",
            aPos="f"
        )
        self.num_instances = 0
        self.default_color = torch.tensor(default_color).float().cuda()
        self.domain.update_buffer("aPos", torch.tensor(np.array(vertices), dtype=torch.float32).cuda())
        self.domain.update_buffer("aNormal", torch.tensor(np.array(normals), dtype=torch.float32).cuda())
    
    @property
    def scale_modifier(self):
        return self.group.scale_modifier
    
    @scale_modifier.setter
    def scale_modifier(self, value):
        self.group.scale_modifier = value
    
    @property
    def sun_direction(self):
        return self.group.sun_direction

    @sun_direction.setter
    def sun_direction(self, value):
        self.group.sun_direction = value


    def resize(self, num_instances: int):
        self.domain.resize(num_instances)
    
    def update_positions(self, positions: torch.Tensor):
        self.resize(positions.shape[0])
        self.domain.update_buffer("position", positions)
    
    def update_quats(self, quats: torch.Tensor):
        # SCALAR FIRST
        self.resize(quats.shape[0])
        self.domain.update_buffer("rotation", quats)
    
    def update_scales(self, scales: torch.Tensor):
        self.resize(scales.shape[0])
        self.domain.update_buffer("scaling", scales)
    
    def update_colors(self, colors: torch.Tensor):
        self.resize(colors.shape[0])
        self.domain.update_buffer("color", colors)

    def update(
        self,
        positions: torch.Tensor,
        rotations: Optional[torch.Tensor] = None, # SCALAR FIRST
        scaling: Optional[torch.Tensor] = None,
        colors: Optional[torch.Tensor] = None,
    ):
        if rotations is None:
            rotations = torch.zeros((positions.shape[0], 4), dtype=torch.float32).cuda()
            rotations[:, 3] = 1.0

        if scaling is None:
            scaling = torch.ones((positions.shape[0], 3), dtype=torch.float32).cuda()

        if colors is None:
            colors = torch.ones((positions.shape[0], 3), dtype=torch.float32).cuda()*self.default_color
        
        self.num_instances = positions.shape[0]
        self.domain.update_buffer("position", positions)
        self.domain.update_buffer("rotation", rotations)
        self.domain.update_buffer("scaling", scaling)
        self.domain.update_buffer("color", colors)

    def draw(self):
        if self.num_instances == 0:
            return
        self.group.set_state()
        self.domain.bind()
        gl.glDrawElementsInstanced(
            gl.GL_TRIANGLES,
            self.index_count,
            gl.GL_UNSIGNED_INT,
            None,
            self.num_instances,
        )
