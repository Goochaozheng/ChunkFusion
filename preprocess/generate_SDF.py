import mesh_to_sdf

import trimesh
from os.path import join
import os
import shutil
import numpy as np
import argparse
from glob import glob

from tqdm import tqdm
from skimage.measure import marching_cubes


"""
Generate SDF volume from ModelNet mesh model.
A model list and a scale list are required:
model_list: list containing path to the mesh model
scale_list: corresponding scale of the mesh model, for scaling the model to real-size

usage: python generate_SDF.py data_list_dir=path/to/model_list/and/scale_list
"""

parser = argparse.ArgumentParser()
# path to scaled ModelNet mesh
parser.add_argument("--data_list_dir", type=str)
parser.add_argument("--chunk_size", type=int, default=16)
parser.add_argument("--voxel_resolution", type=float, default=0.01)
args = parser.parse_args()

listDir = args.data_list_dir
modelList = np.loadtxt(join(listDir, "model_list.txt"), dtype=np.str)
scaleList = np.loadtxt(join(listDir, "scale_list.txt"), dtype=np.float)

tqdm.write("- Voxel Resolution: {}".format(args.voxel_resolution))
tqdm.write("- Chunk Size: {}".format(args.chunk_size))

for i in tqdm(range(len(modelList)), desc="Total Models"):

    # Read origin mesh model
    model = modelList[i]
    scale = scaleList[i]

    model = str.strip(model)
    mesh = trimesh.load(model)
    modelName = model.split("/")[-1].split(".")[0]
    tqdm.write("\n- Reading Model: {}".format(modelName))

    # Scale ModelNet model to actual size (approximately)
    mesh.apply_scale(1./scale)
    # randomly rotate mesh
    rotation = trimesh.transformations.rotation_matrix(
        angle=np.radians(np.random.uniform(0,360)),
        direction=np.random.rand(3),
        point=[0,0,0]
    )
    mesh.apply_transform(rotation)

    mesh.vertices -= mesh.bounding_box.centroid
    tqdm.write("- Mesh Size: {}".format(mesh.bounding_box_oriented.primitive.extents))

    # Extract voxel SDF from mesh
    voxels = mesh_to_sdf.mesh_to_voxels(
        mesh, 
        chunk_size=args.chunk_size,
        voxel_resolution=args.voxel_resolution, 
        surface_point_method='scan',
        sign_method='depth'
    )

    if np.all(voxels > 0) or np.all(voxels < 0):
        print("SDF Error, all SDF > 0")
        continue

    # Generate voxelized mesh and scale to actual size
    vertices, faces, normals, _ = marching_cubes(voxels, level=0)
    # Move mesh center to origin
    vertices -= np.array(voxels.shape) / 2
    vox_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    vox_mesh.apply_scale(args.voxel_resolution)

    # Save SDF value and voxelized mesh
    outputDir = join(listDir, modelName)
    if not os.path.isdir(outputDir):
        os.mkdir(outputDir)

    mesh.export(os.path.join(outputDir, "original_mesh.off"))
    vox_mesh.export(os.path.join(outputDir, "voxelized_mesh.off"))
    np.save(os.path.join(outputDir, "voxel_sdf.npy"), voxels)
