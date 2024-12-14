import ctypes
import math
import numpy as np
import torch


PASTEL_COLORS = torch.tensor(
    [
        [0.984375, 0.7265625, 0.0703125],
        [0.7265625, 0.23046875, 0.23046875],
        [0.23046875, 0.7265625, 0.23046875],
        [0.23046875, 0.23046875, 0.7265625],
        [0.7265625, 0.23046875, 0.7265625],
        [0.23046875, 0.7265625, 0.7265625],
        [0.7265625, 0.7265625, 0.23046875],
        [0.7265625, 0.7265625, 0.7265625],
    ]
).cuda()

COLORS = {
    "BLUE": torch.tensor([0.0, 0.5843, 1.0]),
    "YELLOW": torch.tensor([1.0, 0.8, 0.0]),
    "RED": torch.tensor([0.835, 0.368, 0.0]),
    "GREEN": torch.tensor([0.008, 0.6186, 0.45098]),
    "PURPLE": torch.tensor([0.51372, 0.298, 0.49012]),
    "ORANGE": torch.tensor([1.0, 0.4, 0.0]),
}


def str_buffer(string: str):
    return ctypes.c_char_p(string.encode("utf-8"))

def convert_K_to_projection_matrixT(
    K: np.ndarray, width: int, height: int, near: float = 0.01, far: float = 10.0
) -> np.ndarray:
    fx, fy, u0, v0 = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    l = -u0 * near / fx
    r = (width - u0) * near / fx
    b = -(height - v0) * near / fy
    t = v0 * near / fy

    return np.array(
        [
            [2 * near / (r - l), 0, (r + l) / (r - l), 0],
            [0, 2 * near / (t - b), (t + b) / (t - b), 0],
            [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
            [0, 0, -1.0, 0],
        ],
        dtype=np.float32,
    ).T

def calibration_matrix_values(camera_matrix: np.ndarray, image_size: tuple[int, int], aperture_width: float=0.0, aperture_height: float=0.0):
    # https://github.com/opencv/opencv/blob/93b607dc72e1d7953b17a58d8fb5f130b05c3d7a/modules/calib3d/src/calibration.cpp#L3988
    if camera_matrix.shape != (3, 3):
        raise ValueError("Size of camera_matrix must be 3x3!")
    
    K = camera_matrix
    assert image_size[0] != 0 and image_size[1] != 0 and K[0, 0] != 0.0 and K[1, 1] != 0.0

    # Calculate pixel aspect ratio
    aspect_ratio = K[1, 1] / K[0, 0]

    # Calculate number of pixel per realworld unit
    if aperture_width != 0.0 and aperture_height != 0.0:
        mx = image_size[0] / aperture_width
        my = image_size[1] / aperture_height
    else:
        mx = 1.0
        my = aspect_ratio

    # Calculate fovx and fovy
    fovx = math.atan2(K[0, 2], K[0, 0]) + math.atan2(image_size[0] - K[0, 2], K[0, 0])
    fovy = math.atan2(K[1, 2], K[1, 1]) + math.atan2(image_size[1] - K[1, 2], K[1, 1])
    fovx = fovx * 180.0 / math.pi
    fovy = fovy * 180.0 / math.pi

    # Calculate focal length
    focal_length = K[0, 0] / mx

    # Calculate principle point
    principal_point = (K[0, 2] / mx, K[1, 2] / my)

    return {
        'fovx': fovx,
        'fovy': fovy,
        'focal_length': focal_length,
        'principal_point': principal_point,
        'aspect_ratio': aspect_ratio
    }

DEFAULT_WIDTH= 1280
DEFAULT_HEIGHT = 720
DEFAULT_F = 652
DEFAULT_K = np.array([[DEFAULT_F, 0, 640], [0, DEFAULT_F, 360], [0, 0, 1]], dtype=np.float32)
DEFAULT_K_OPENGL_T = convert_K_to_projection_matrixT(DEFAULT_K, DEFAULT_WIDTH, DEFAULT_HEIGHT)