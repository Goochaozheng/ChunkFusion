import torch
import trimesh
import numpy as np
from skimage.measure import marching_cubes

def pixelToPoint(depth:torch.Tensor, intrinsics:torch.Tensor):
    """
    Convert pixel with depth measurement to 3D point.
    Only valid depth measurements are returned.

    :param depth: Tensor, depth image 
    :param intrinsics: Tensor, (3 x 3) intrinsic matrix
    :return: 3D points, (N x 3), N is num of valid depth measurements.
    """

    with torch.no_grad():

        h, w = depth.shape
        device = depth.device

        u, v = torch.meshgrid(
            torch.arange(w),
            torch.arange(h),
        )

        pixels = torch.cat([
            u.reshape(1, -1),
            v.reshape(1, -1),
            torch.ones((1, h*w))
        ], axis=0).to(device).T

        depth = depth.T.flatten()
        validMask = (depth != 0)
        validDepth = depth[validMask].repeat(3, 1).T
        validPixels = pixels[validMask]

        points = (intrinsics.inverse() @ validPixels.T).T * validDepth

    return points


def pointToPixel(points:torch.Tensor, intrinsics:torch.Tensor):
    """
    Project 3D points to camera pixels.
    
    :param point: Tensor, 3D points (N x 3)
    :param intrinsics: Tensor, (3 x 3) intrinsic matrix
    :return: pixels coords (N x 2), [u, v]
    """

    with torch.no_grad():
        pointDepth = points[:, 2].repeat(3, 1).T
        pixels = torch.round((points @ intrinsics.T ) / pointDepth)
    return pixels


def transformPoints(points:torch.Tensor, transform:torch.Tensor):
    """
    Apply ridgid transform on point clouds

    :param transform: Tensor, (4 x 4) transform matrix
    """ 
    with torch.no_grad():
        points_h = torch.cat([points, torch.ones((points.shape[0], 1), device=points.device)], 1) 
        points_h = (transform @ points_h.T).T
    return points_h[:, :3]


def showMesh(volume):
    mask = np.abs(volume) < 1
    # volume[mask] = np.nan
    try:
        vertices, faces, normals, _ = marching_cubes(volume, level=0, mask=mask)
        mesh = trimesh.Trimesh(vertices, faces, normals)
        mesh.show()
    except Exception:
        print("No surface found")