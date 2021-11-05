import torch
import numpy as np
from geometry.bbox import BBox

class Chunk:

    def __init__(self, chunkID, chunkSize:int, voxelResolution:float, device):
        """
        Voxel chunk, a grid volume containing chunkSize^3 voxels.

        :param chunkID: Tuple, 3-dim int index of chunk.
        :param chunkSize: Int, Size of the voxel chunk, total num of voxels is chunkSize^3.
        :param voxelResolution: Float, The resolution of single voxel, in meter.
        :param device: torch device
        """

        self.chunkSize = chunkSize
        self.voxelResolution = voxelResolution
        self.chunkShape = (chunkSize, chunkSize, chunkSize)
        self.device = device

        self.chunkID = chunkID
        self.chunkOrigin = np.array(self.chunkID) * chunkSize * voxelResolution

        # Actual data stored in the chunk
        self.tsdfValues = torch.ones((self.chunkSize, self.chunkSize, self.chunkSize), dtype=torch.float32, device=self.device, requires_grad=False)
        self.weightValues = torch.zeros((self.chunkSize, self.chunkSize, self.chunkSize), dtype=torch.float32, device=self.device, requires_grad=False)
        self.obsCount = 0

    def reset(self):
        self.tsdfValues = torch.ones((self.chunkSize, self.chunkSize, self.chunkSize), dtype=torch.float32, device=self.device, requires_grad=False)
        self.weightValues = torch.zeros((self.chunkSize, self.chunkSize, self.chunkSize), dtype=torch.float32, device=self.device, requires_grad=False)
        self.obsCount = 0
            

    def getChunkTSDF(self):
        return self.tsdfValues

    def getChunkWeight(self):
        return self.weightValues

    def getObsCount(self):
        return self.obsCount

    def hasVoxel(self, x: int, y: int, z: int) -> bool:
        return 0 <= x < self.chunkSize and 0 <= y < self.chunkSize and 0 <= z < self.chunkSize

    def getChunkGeometry(self, color):
        bboxMin = self.chunkOrigin
        bboxMax = self.chunkOrigin + self.chunkSize * self.voxelResolution
        return BBox(bboxMin, bboxMax).getBboxGeometry(color)
