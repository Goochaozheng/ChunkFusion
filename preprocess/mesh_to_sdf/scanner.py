from .pyrender_wrapper import CustomShaderCache
from . import pyrender_wrapper
from .utils import get_equidistant_camera_angles, get_camera_transform_looking_at_origin, get_random_uniform_camera_angles
from .surface_point_cloud import SurfacePointCloud
from .scan import Scan

import pyrender
import random
import math
import numpy as np
from tqdm import tqdm



class Scanner:
    """
    Class for creating surface point cloud with depth scan or mesh sampling.
    """

    def __init__(self, mesh, bounding_radius) -> None:
        self.bounding_radius = bounding_radius
        self.scene = pyrender.Scene()
        self.mesh = mesh
        self.mesh_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(mesh), matrix=np.eye(4))
        self.scene.add_node(self.mesh_node)

    def _set_perspective_camera(self, y_fov, aspect_ratio, z_near, z_far, scan_resolution):
        self.camera = pyrender.PerspectiveCamera(yfov=y_fov, aspectRatio=aspect_ratio, znear=z_near, zfar=z_far)
        self.camera_node = pyrender.Node(camera=self.camera, matrix=np.eye(4))
        self.scene.add_node(self.camera_node)

        self.renderer = pyrender.OffscreenRenderer(scan_resolution, scan_resolution)
        self.renderer._renderer._program_cache = CustomShaderCache()

    def _set_intrinsic_camera(self, fx, fy, cx, cy, z_near, z_far, scan_width, scan_height):
        self.camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=z_near, zfar=z_far)
        self.camera_node = pyrender.Node(camera=self.camera, matrix=np.eye(4))
        self.scene.add_node(self.camera_node)

        self.renderer = pyrender.OffscreenRenderer(scan_width, scan_height)
        self.renderer._renderer._program_cache = CustomShaderCache()

    def create_surface_points_from_scans(self, y_fov, z_near, z_far, scan_count, scan_resolution, calculate_normals=True):
        """
        Create surface point cloud by scanning surround the model.
        For SDF computation, point cloud for closest point querying and depth value for determing sign of SDF.
        """
        
        # Surface scan is based on perspective camera for better scan coverage.
        self._set_perspective_camera(y_fov=y_fov, aspect_ratio=1, z_near=z_near, z_far=z_far, scan_resolution=scan_resolution)
        camera_distance = self.bounding_radius + 1.0 if self.bounding_radius > 1.0 else 2 * self.bounding_radius
        scans = []
        for phi, theta in get_equidistant_camera_angles(scan_count):

            camera_transform = get_camera_transform_looking_at_origin(phi, theta, camera_distance=camera_distance)
            color, depth = self.render_normal_and_depth_buffers(camera_transform)

            depthScan = Scan(
                color=color,
                depth=depth,
                camera_transform=camera_transform,
                projection_matrix=self.camera.get_projection_matrix(),
                resolution=scan_resolution,
                z_near=z_near,
                z_far=z_far,
                calculate_normals=calculate_normals
            )

            scans.append(depthScan)

        return SurfacePointCloud(self.mesh, 
            points=np.concatenate([scan.points for scan in scans], axis=0),
            normals=np.concatenate([scan.normals for scan in scans], axis=0) if calculate_normals else None,
            scans=scans
        )


    def create_surface_points_from_sample(self, sample_point_count=10000000, calculate_normals=True):
        if calculate_normals:
            points, face_indices = self.mesh.sample(sample_point_count, return_index=True)
            normals = self.mesh.face_normals[face_indices]
        else:
            points = self.mesh.sample(sample_point_count, return_index=False)

        return SurfacePointCloud(self.mesh, 
            points=points,
            normals=normals if calculate_normals else None,
            scans=None
    )


    def create_random_depth_scan(self, fx, fy, cx, cy, z_near, z_far, scan_width, scan_height, scan_count=1000):
        """
        Create random depth scan surround the model based on an intrinsic camera.
        For depth image generation.
        """

        # depth image scan is based on intrinsics camera for better parameter handling.
        self._set_intrinsic_camera(fx, fy, cx, cy, z_near, z_far, scan_width, scan_height)

        # generate random scan
        for phi, theta in get_random_uniform_camera_angles(scan_count):

            distance_random_bias = random.uniform(-0.15, 0.15)
            
            if self.bounding_radius > 1.:
                camera_distance = self.bounding_radius + 1.0
            else:
                camera_distance = 2 * self.bounding_radius 
            
            camera_distance = camera_distance + distance_random_bias
            camera_transform = get_camera_transform_looking_at_origin(phi, theta, camera_distance)

            # Get depth scan
            color, depth = self.render_normal_and_depth_buffers(camera_transform)

            yield color, depth, camera_transform



    def render_normal_and_depth_buffers(self, camera_transform):

        pyrender_wrapper.suppress_multisampling = True

        self.scene.set_pose(self.camera_node, pose=camera_transform)

        color, depth = self.renderer.render(self.scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)

        pyrender_wrapper.suppress_multisampling = False

        return color, depth
