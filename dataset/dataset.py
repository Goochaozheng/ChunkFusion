import cv2
from numpy.core.fromnumeric import size
import trimesh
import torch
import h5py
import numpy as np

from os.path import join
from torch.utils.data import Dataset
from glob import glob

class ModelDataset(Dataset):
    """
    Dataset for loading single model.
    """

    def __init__(self, dataDir, dataList, depthScale, frameNum, noiseScan, shuffle, useMesh, device) -> None:
        super().__init__()
        self.dataDir = dataDir
        self.dataList = dataList
        self.device = device

        self.depthScale = depthScale
        self.frameNum = frameNum
        self.noiseScan = noiseScan
        self.useMesh = useMesh
        self.shuffle = shuffle
        
        self.modelList = np.loadtxt(join(self.dataDir, self.dataList), dtype=str)

        self.modelList = np.sort(self.modelList)
        if self.shuffle:
            np.random.shuffle(self.modelList)

    def __len__(self):
        return len(self.modelList)

    def __getitem__(self, index):
        modelDir = self.modelList[index]
        modelName = modelDir.split("/")[-2]
        voxelSDF = np.load(join(self.dataDir, modelDir, "voxel_sdf.npy"))
        voxelSDF = torch.from_numpy(voxelSDF).to(self.device)

        frameDataset = FrameDataset(
            frameDir=join(self.dataDir, modelDir, "scan"), 
            depthScale=self.depthScale,
            frameNum=self.frameNum, 
            noiseScan=self.noiseScan, 
            shuffle=self.shuffle, 
            device=self.device
        )
        voxelizedMesh = None

        if self.useMesh:
            voxelizedMesh = trimesh.load(join(self.dataDir, modelDir, "voxelized_mesh.off"))
        
        return {
            "voxelSDF": voxelSDF, 
            "frameDataset": frameDataset, 
            "voxelizedMesh": voxelizedMesh, 
            "modelName": modelName
        }
        

    def shuffleData(self):
        np.random.shuffle(self.modelList)


    def getRandomModel(self):
        idx = int(np.random.rand(1) * len(self.modelList))
        modelDir = self.modelList[idx]
        modelName = modelDir.split("/")[-2]
        voxelSDF = np.load(join(self.dataDir, modelDir, "voxel_sdf.npy"))
        voxelSDF = torch.from_numpy(voxelSDF).to(self.device)

        frameDataset = FrameDataset(
            frameDir=join(self.dataDir, modelDir, "scan"), 
            depthScale=self.depthScale,
            frameNum=self.frameNum, 
            noiseScan=self.noiseScan, 
            shuffle=self.shuffle, 
            device=self.device
        )
        voxelizedMesh = None

        if self.useMesh:
            voxelizedMesh = trimesh.load(join(self.dataDir, modelDir, "voxelized_mesh.off"))
        
        return {
            "voxelSDF": voxelSDF, 
            "frameDataset": frameDataset, 
            "voxelizedMesh": voxelizedMesh, 
            "modelName": modelName
        }       



class FrameDataset(Dataset):
    """
    Dataset for loading depth scans of single model
    """

    def __init__(self, frameDir, depthScale, frameNum, noiseScan, shuffle, device) -> None:
        super().__init__()
        self.frameDir = frameDir
        self.depthScale = depthScale
        self.frameNum = frameNum
        self.device = device
        
        if noiseScan:
            frameList = glob(join(self.frameDir, "[0-9]*.depth.noise.png"))
        else:
            frameList = glob(join(self.frameDir, "[0-9]*.depth.png"))
        
        poseList = glob(join(self.frameDir, "[0-9]*.pose.txt"))

        assert len(frameList) == len(poseList)

        self.frameList = np.sort(frameList)[:frameNum]
        self.poseList = np.sort(poseList)[:frameNum]
        self.dataList = np.c_[self.frameList, self.poseList]

        if shuffle:
            np.random.shuffle(self.dataList) 


    def getIntrinsics(self):
        intrinsic = np.loadtxt(join(self.frameDir, "intrinsic.txt"))
        intrinsic = torch.from_numpy(intrinsic).to(self.device).float()
        return intrinsic

    def __len__(self):
        return self.frameNum

    def __getitem__(self, index):
        imgDir = self.dataList[index][0]
        poseDir = self.dataList[index][1]

        frameIdx = imgDir.split('/')[-1].split('.')[0]
        poseIdx = poseDir.split('/')[-1].split('.')[0]
        assert frameIdx == poseIdx

        depthImg = cv2.imread(imgDir, cv2.IMREAD_ANYDEPTH).astype(float)
        depthImg /= self.depthScale

        cameraPose = np.loadtxt(poseDir)

        return {
            "depthImg": depthImg, 
            "cameraPose": cameraPose
        }


class ChunkDataset(Dataset):
    """
    Dataset for loading chunk pairs
    """

    def __init__(self, dataDir, device) -> None:
        super().__init__()

        self.dataDir = dataDir
        self.device = device

        self.gtData = np.load(join(self.dataDir, "chunk_gt.npy"))
        self.inputData = np.load(join(self.dataDir, "chunk_input.npy"))

        assert len(self.gtData) == len(self.inputData)

        self.indexRange = np.arange(self.__len__())


    def __len__(self):        
        return len(self.gtData)


    def __getitem__(self, index):

        # if self.gtData == None:
        #     self.gtData = h5py.File(join(self.dataDir, "chunk_gt.h5"), 'r')["chunk_gt"]

        # if self.inputData == None:
        #     self.inputData = h5py.File(join(self.dataDir, "chunk_input.h5"), 'r')["chunk_input"]
        
        input = self.inputData[index]
        gt = self.gtData[index]

        return {
            "input": input,
            "gt": gt
        }


    def getRandomBatch(self, size):
        # if self.gtData == None:
        #     self.gtData = h5py.File(join(self.dataDir, "chunk_gt.h5"), 'r')["chunk_gt"]

        # if self.inputData == None:
        #     self.inputData = h5py.File(join(self.dataDir, "chunk_input.h5"), 'r')["chunk_input"]

        index = np.sort(np.random.choice(self.indexRange, size=size, replace=False))

        input = torch.from_numpy(self.inputData[index]).to("cuda:0")
        gt = torch.from_numpy(self.gtData[index]).to("cuda:0")

        return {
            "input": input,
            "gt": gt
        }