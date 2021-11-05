from mesh_to_sdf.scanner import Scanner
from mesh_to_sdf.utils import get_rotation_matrix
from simkinect import add_noise

import trimesh
import os
import glob
import numpy as np
import argparse
import cv2

from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str)
parser.add_argument("--scan_count", type=int, default=200)
parser.add_argument("--scan_width", type=int, default=640)
parser.add_argument("--scan_height", type=int, default=480)
parser.add_argument("--depth_min", type=float, default=0.1)
parser.add_argument("--depth_max", type=float, default=6.0)
parser.add_argument("--focal_length", type=float, default=500)
parser.add_argument("--baseline_m", type=float, default=0.075)

args = parser.parse_args()

modelDir = glob.glob(os.path.join(args.data_dir, "[a-z]*_[0-9]*/"))

dot_pattern_ = cv2.imread("preprocess/simkinect/kinect-pattern_3x3.png", 0)
invalid_disp_ = 99999999.9

for dir in tqdm(modelDir, desc="Total Models"):

    outputDir = os.path.join(dir, "scan")
    if not os.path.isdir(outputDir):
        os.mkdir(outputDir)

    tqdm.write("\n- Reading Model: {}".format(dir))
    modelName = dir.split("/")[-1]
    mesh = trimesh.load(os.path.join(dir, "original_mesh.off"))

    boundingRadius = np.max(np.linalg.norm(mesh.bounding_box.vertices, axis=1)) * 1.1

    scanner = Scanner(mesh, boundingRadius)

    # write intrinsics to file
    intrinsicMatrix = np.array([
        [args.focal_length, 0., args.scan_width/2], 
        [0., args.focal_length, args.scan_height/2],
        [0., 0., 1.]
    ])

    np.savetxt(os.path.join(outputDir, "intrinsic.txt"), intrinsicMatrix)

    tqdm.write("- Generating Scan ...")

    index = 0

    for color, depth, camera_transform in tqdm(scanner.create_random_depth_scan(
        fx=args.focal_length, fy=args.focal_length, cx=args.scan_width/2, cy=args.scan_height/2, z_near=args.depth_min, 
        z_far=args.depth_max, scan_width=args.scan_width, scan_height=args.scan_height, scan_count=args.scan_count
    ), total=args.scan_count):

        # Save origin depth scan
        depth_img = (depth * 1000).astype(np.uint16)
        color_img = (color).astype(np.uint8)
        cv2.imwrite(os.path.join(outputDir, "{:06d}.depth.png".format(index)), depth_img)
        cv2.imwrite(os.path.join(outputDir, "{:06d}.color.png".format(index)), color_img)

        # Add noise and artifact on the scan result
        h, w = depth.shape
        depth_interp = add_noise.add_gaussian_shifts(depth)

        disp_= args.focal_length * args.baseline_m / (depth_interp + 1e-10)
        depth_f = np.round(disp_ * 8.0)/8.0

        out_disp = add_noise.filterDisp(depth_f, dot_pattern_, invalid_disp_)

        depth = args.focal_length * args.baseline_m / out_disp
        depth[out_disp == invalid_disp_] = 0 

        noisy_depth = np.zeros_like(depth)
        noisy_depth[depth!=0] = (35130/np.round((35130/np.round(depth[depth!=0]*100)) + np.random.normal(size=depth[depth!=0].shape)*(1.0/6.0) + 0.5))/100 

        depth_noise_img = (noisy_depth * 1000).astype(np.uint16)
        cv2.imwrite(os.path.join(outputDir, "{:06d}.depth.noise.png".format(index)), depth_noise_img)

        # pyrender use openGL camera coordinates, z-axis pointing away from the view frustum and y pointing up
        # convert to coordinates with z-axis pointing front, toward the frustum and y-axis pointing down
        camera_transform = np.matmul(camera_transform, get_rotation_matrix(np.pi, axis='y'))
        camera_transform = np.matmul(camera_transform, get_rotation_matrix(np.pi, axis='z'))
        np.savetxt(os.path.join(outputDir, "{:06d}.pose.txt".format(index)), camera_transform)

        # Add noise to camera transform
        camera_transform_noise = np.copy(camera_transform)
        camera_translation = camera_transform[:3, 3]
        camera_translation_bias = (np.random.rand(3) - 0.5) * 0.05 * camera_translation
        camera_transform_noise[:3, 3] += camera_translation_bias
        np.savetxt(os.path.join(outputDir, "{:06d}.pose.noise.txt".format(index)), camera_transform_noise)
        
        index += 1