import cv2
import torch
import h5py
import argparse
import shutil
import numpy as np
from tqdm import tqdm
from glob import glob
from os.path import join
from skimage.measure import marching_cubes

import sys,os
sys.path.insert(0, os.path.join(os.getcwd()))

from module import FusionIntegrator, constructChunksFromVolume
from geometry import DepthPointCloud

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str)
parser.add_argument("--out_dir", type=str)
parser.add_argument("--num_frame", type=int, default=20)

args = parser.parse_args()

if __name__ == "__main__":
  
    modelList = glob(join(args.data_dir, "[a-z]*_[0-9]*"))

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    h5_input = h5py.File(join(args.out_dir, "chunk_input.h5"), "w")
    inputDataset = h5_input.create_dataset("chunk_input", shape=(0, 4096), maxshape=(None, 4096), dtype='f', chunks=(1,4096))

    h5_gt = h5py.File(join(args.out_dir, "chunk_gt.h5"), "w")
    gtDataset = h5_gt.create_dataset("chunk_gt", shape=(0, 4096), maxshape=(None, 4096), dtype='f', chunks=(1,4096))

    useGPU = True if torch.cuda.is_available() else False
    device = torch.device("cuda" if useGPU else "cpu")
    torch.cuda.set_device(0)

    inputData = torch.empty((0, 4096), device=device)
    gtData = torch.empty((0, 4096), device=device)

    for modelIdx in range(len(modelList)):
    
        modelDir = modelList[modelIdx]
        print("Loading Model [{}/{}] : {}".format(modelIdx, len(modelList), modelDir))

        integrator = FusionIntegrator(
            chunkSize=16,
            voxelResolution=0.01,
            truncation=0.04,
            minPointsPerChunk=50,
            meshing=False,
            withBestScan=False,
            withFuser=False,
            withParser=False,
            parserModel=None,
            fuserModel=None,
            batchSize=512,
            padding=0,
            device=device,
            verbose=False
        )

        gtVolume = np.load(join(modelDir, "voxel_sdf.npy"))
        chunkManager_gt = constructChunksFromVolume(
            torch.from_numpy(gtVolume).to(device),
            chunkSize=16,
            voxelResolution=0.01,
            truncation=0.04,
            minPointsPerChunk=50,
            meshing=False,
            device=device
        )

        intrinsics = np.loadtxt(join(modelDir, "scan", "intrinsic.txt"))

        frameList = np.sort(glob(join(modelDir, "scan", "*.depth.noise.png")))
        poseList = np.sort(glob(join(modelDir, "scan", "*.pose.txt")))
        dataList = np.c_[frameList, poseList]
        np.random.shuffle(dataList)

        for index in tqdm(range(args.num_frame)):

            imgDir = dataList[index][0]
            poseDir = dataList[index][1]

            frameIdx = imgDir.split('/')[-1].split('.')[0]
            poseIdx = poseDir.split('/')[-1].split('.')[0]
            assert frameIdx == poseIdx

            depthImg = cv2.imread(imgDir, cv2.IMREAD_ANYDEPTH).astype(float)
            depthImg /= 1000.

            pose = np.loadtxt(poseDir)

            frame = DepthPointCloud(
                torch.from_numpy(depthImg).to(device),
                torch.from_numpy(intrinsics).to(device),
                torch.from_numpy(pose).to(device)
            )

            chunkList, _ = integrator.getChunkFromPointCloud(frame)

            voxelPoints = integrator.getChunkListData(chunkList, withPad=False)["voxelPoints"]

            inputTSDF = integrator.computeTSDF(frame, voxelPoints)

            TSDFGroudTruth = chunkManager_gt.getChunkListData(chunkList, withPad=False)["voxelTSDF"]

            assert len(inputTSDF) == len(TSDFGroudTruth)

            inputTSDF = inputTSDF.reshape((-1, 4096))
            TSDFGroudTruth = TSDFGroudTruth.reshape((-1, 4096))

            inputData = torch.cat((inputData, inputTSDF), dim=0)
            gtData = torch.cat((gtData, TSDFGroudTruth), dim=0)

        print("Chunk Count: {}".format(len(inputData)))

        indexBegin = inputDataset.shape[0]
        indexEnd = indexBegin + len(inputData)

        inputDataset.resize((indexEnd , 4096))
        inputDataset[indexBegin : indexEnd] = inputData.cpu().numpy()

        gtDataset.resize((indexEnd, 4096))
        gtDataset[indexBegin : indexEnd] = gtData.cpu().numpy()

    print("Saving numpy array")

    np.save(join(args.out_dir, "chunk_input.npy"), h5_input["chunk_input"])
    np.save(join(args.out_dir, "chunk_gt.npy"), h5_gt["chunk_gt"])

    h5_input.close()
    h5_gt.close()


