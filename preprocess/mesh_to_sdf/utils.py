import functools
from networkx.algorithms.cuts import volume
from scipy.spatial.transform import Rotation
import trimesh
import math
import numpy as np


def get_scan_count(mesh):    
    meshSize = mesh.bounding_box_oriented.primitive.extents
    count = int(np.mean(meshSize) * 128)
    return max(count, 128)

def get_scan_resolution(mesh):
    meshSize = mesh.bounding_box_oriented.primitive.extents
    resolution = int(np.mean(meshSize) * 640)
    return min(max(resolution, 640), 1000)


def get_volume_size(mesh, chunk_size, voxel_resolution):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()
    
    # Move the bounding box to center
    mesh.vertices = mesh.vertices - mesh.bounding_box.centroid
    # expand the volume to cover the model and make sure the volume center aligns to the origin
    half_size = np.ceil(mesh.bounding_box.primitive.extents*1.2 / (2 * voxel_resolution))
    
    # Round volumes size to match chunk size
    temp = half_size[half_size % chunk_size != 0] 
    new = np.ceil(temp / chunk_size) * chunk_size
    half_size[half_size % chunk_size != 0] = new

    volume_size = 2 * half_size
    return volume_size.astype(int)


def scale_to_unit_sphere(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    distances = np.linalg.norm(vertices, axis=1)
    vertices /= np.max(distances)

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

def scale_to_unit_cube(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    vertices *= 2 / np.max(mesh.bounding_box.extents)

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

# Use get_raster_points.cache_clear() to clear the cache
# @functools.lru_cache(maxsize=4)
def get_raster_points(volume_size, voxel_resolution):
    ID_x, ID_y, ID_z = np.meshgrid(
        np.arange(-volume_size[0]/2, volume_size[0]/2),
        np.arange(-volume_size[1]/2, volume_size[1]/2),
        np.arange(-volume_size[2]/2, volume_size[2]/2),
    indexing='ij')

    points = np.concatenate([
        ID_x.reshape(1, -1),
        ID_y.reshape(1, -1),
        ID_z.reshape(1, -1)
    ], axis=0).astype(float).T

    return (points + 0.5) * voxel_resolution

def check_voxels(voxels):
    block = voxels[:-1, :-1, :-1]
    d1 = (block - voxels[1:, :-1, :-1]).reshape(-1)
    d2 = (block - voxels[:-1, 1:, :-1]).reshape(-1)
    d3 = (block - voxels[:-1, :-1, 1:]).reshape(-1)

    max_distance = max(np.max(d1), np.max(d2), np.max(d3))
    return max_distance < 2.0 / voxels.shape[0] * 3**0.5 * 1.1

def sample_uniform_points_in_unit_sphere(amount):
    unit_sphere_points = np.random.uniform(-1, 1, size=(amount * 2 + 20, 3))
    unit_sphere_points = unit_sphere_points[np.linalg.norm(unit_sphere_points, axis=1) < 1]

    points_available = unit_sphere_points.shape[0]
    if points_available < amount:
        # This is a fallback for the rare case that too few points are inside the unit sphere
        result = np.zeros((amount, 3))
        result[:points_available, :] = unit_sphere_points
        result[points_available:, :] = sample_uniform_points_in_unit_sphere(amount - points_available)
        return result
    else:
        return unit_sphere_points[:amount, :]


def get_equidistant_camera_angles(count):
    increment = math.pi * (3 - math.sqrt(5))
    for i in range(count):
        theta = math.asin(-1 + 2 * i / (count - 1))
        phi = ((i + 1) * increment) % (2 * math.pi)
        yield phi, theta


def get_random_uniform_camera_angles(count):
    u = np.random.rand(count)
    v = np.random.rand(count)
    for i in range(count):
        theta = 2 * math.pi * u[i]
        phi = math.acos(2 * v[i] - 1)
        yield phi, theta


def get_rotation_matrix(angle, axis='y'):
    matrix = np.identity(4)
    if hasattr(Rotation, "as_matrix"): # scipy>=1.4.0
        matrix[:3, :3] = Rotation.from_euler(axis, angle).as_matrix()
    else: # scipy<1.4.0
        matrix[:3, :3] = Rotation.from_euler(axis, angle).as_dcm()
    return matrix

def get_camera_transform_looking_at_origin(rotation_y, rotation_x, camera_distance=2):
    camera_transform = np.identity(4)
    camera_transform[2, 3] = camera_distance
    camera_transform = np.matmul(get_rotation_matrix(rotation_x, axis='x'), camera_transform)
    camera_transform = np.matmul(get_rotation_matrix(rotation_y, axis='y'), camera_transform)
    return camera_transform