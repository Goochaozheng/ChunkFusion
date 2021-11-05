from mesh_to_sdf.scanner import Scanner
from . import utils 

import numpy as np
from tqdm.std import tqdm
import trimesh

def mesh_to_voxels(mesh, chunk_size=16, voxel_resolution=0.02, sign_method='depth', surface_point_method='scan', normal_sample_count=11, pad=False, check_result=False, return_gradients=False):
    """
    Compute the voxel SDF from the mesh.
    The voxel volume is constructed according to the bounding box of the model.
    
    :param mesh: Input mesh
    :param voxel_resolution: Resolution of single voxel, in meter.
    :param surface
    """

    surface_point_cloud = get_surface_point_cloud(mesh, surface_point_method, calculate_normals=sign_method=='normal')

    volume_size = utils.get_volume_size(mesh, chunk_size=chunk_size, voxel_resolution=voxel_resolution)
    tqdm.write("- Volume Size: {}".format(volume_size))

    voxels = surface_point_cloud.get_voxels(volume_size, voxel_resolution, sign_method=='depth', normal_sample_count, pad, check_result, return_gradients)

    return voxels


def mesh_to_sdf(mesh, query_points, surface_point_method='scan', sign_method='normal', bounding_radius=None, scan_count=100, scan_resolution=400, sample_point_count=10000000, normal_sample_count=11):
    if not isinstance(query_points, np.ndarray):
        raise TypeError('query_points must be a numpy array.')
    if len(query_points.shape) != 2 or query_points.shape[1] != 3:
        raise ValueError('query_points must be of shape N ✕ 3.')
    
    if surface_point_method == 'sample' and sign_method == 'depth':
        print("Incompatible methods for sampling points and determining sign, using sign_method='normal' instead.")
        sign_method = 'normal'

    point_cloud = get_surface_point_cloud(mesh, surface_point_method, scan_count, scan_resolution, sample_point_count, calculate_normals=sign_method=='normal')

    if sign_method == 'normal':
        return point_cloud.get_sdf_in_batches(query_points, use_depth_buffer=False)
    elif sign_method == 'depth':
        return point_cloud.get_sdf_in_batches(query_points, use_depth_buffer=True, sample_count=sample_point_count)
    else:
        raise ValueError('Unknown sign determination method: {:s}'.format(sign_method))



def sample_sdf_near_surface(mesh, number_of_points = 500000, surface_point_method='scan', sign_method='normal', scan_count=100, scan_resolution=400, sample_point_count=10000000, normal_sample_count=11, min_size=0, return_gradients=False):
    """
    Sample some uniform points and some normally distributed around the surface as proposed in the DeepSDF paper
    """
    mesh = utils.scale_to_unit_sphere(mesh)
    
    if surface_point_method == 'sample' and sign_method == 'depth':
        print("Incompatible methods for sampling points and determining sign, using sign_method='normal' instead.")
        sign_method = 'normal'

    surface_point_cloud = get_surface_point_cloud(mesh, surface_point_method, 1, scan_count, scan_resolution, sample_point_count, calculate_normals=sign_method=='normal' or return_gradients)

    return surface_point_cloud.sample_sdf_near_surface(number_of_points, surface_point_method=='scan', sign_method, normal_sample_count, min_size, return_gradients)


def get_surface_point_cloud(mesh, surface_point_method='scan', calculate_normals=False):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("The mesh parameter must be a trimesh mesh.")

    bounding_radius = np.max(np.linalg.norm(mesh.bounding_box.vertices, axis=1)) * 1.1
    scanner = Scanner(mesh, bounding_radius=bounding_radius)

    if surface_point_method == 'scan':
        scan_count = utils.get_scan_count(mesh)
        scan_resolution = utils.get_scan_resolution(mesh)
        tqdm.write("- Generating Scan, Count: {}, Resolution: {} ...".format(scan_count, scan_resolution))
        return scanner.create_surface_points_from_scans(y_fov=1.0472, z_near=0.1, z_far=bounding_radius * 4, 
                                                        scan_count=scan_count, scan_resolution=scan_resolution, calculate_normals=calculate_normals)
    
    elif surface_point_method == 'sample': 
        sample_point_count=10000000
        tqdm.write("- Generating Sample, Count: {} ...".format(sample_point_count))
        return scanner.create_surface_points_from_sample(sample_point_count=sample_point_count, calculate_normals=True)  
   
    else:
        raise ValueError('Unknown surface point sampling method: {:s}'.format(surface_point_method))