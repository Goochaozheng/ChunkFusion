import torch
from time import time

from geometry import transformPoints, pointToPixel
from .chunkManager import ChunkManager

class TSDFIntegrator(ChunkManager):

    def __init__(self, chunkSize:int, voxelResolution:float, truncation:float, minPointsPerChunk:int, meshing:bool, device, verbose) -> None:
        ChunkManager.__init__(self, chunkSize, voxelResolution, truncation, minPointsPerChunk, 0, meshing, device)
        self.verbose = verbose


    def integrateFrame(self, depthPointCloud):
        """
        Integration with TSDF.
        """

        with torch.no_grad():

            # Get chunk with depth measurements
            if self.verbose:
                intersect_begin = time()

            chunkList, obsCount = self.getChunkFromPointCloud(depthPointCloud)
            if len(chunkList) == 0:
                print("- No Chunk Intersecting.")
                return

            # if self.verbose:
            #     print("- Get {} chunk intersecting. Timing: {}".format(len(chunkList), time() - intersect_begin))    
            #     allocation_begin = time()

            # Allocate chunks 
            chunkData = self.getChunkListData(chunkList, withPad=False)
            voxelPoints = chunkData["voxelPoints"]
            voxelTSDF = chunkData["voxelTSDF"]
            voxelWeight = chunkData["voxelWeight"]

            # if self.verbose:
            #     print("- Allocate {} chunks. Timing: {}".format(len(chunkList), time() - allocation_begin))    
            #     integrate_begin = time()

            # Integrate frame into current voxel TSDF
            voxelTSDF, voxelWeight = self.tsdfUpdate(
                depthPointCloud,
                voxelPoints, 
                voxelTSDF, 
                voxelWeight, 
                self.truncation, 
                self.chunkSize
            )

            # Update TSDF value into chunk manager
            self.setChunkListData(chunkList, voxelTSDF, voxelWeight)

            if self.verbose:
                print("- Integrating {} chunks. Timing: {:.06f}".format(len(chunkList), time() - intersect_begin))    
                # print("- Total time: {}".format(time() - intersect_begin))    

        return chunkList



    def tsdfUpdate(self, frame, voxelPoints, voxelTSDF, voxelWeight, truncation, chunkSize):
        """
        TSDF Updating.

        :param frame: DepthPointCloud of current frame, containing depth map, intrinsics & pose.
        :param voxelPoints: 3D coordinates of all voxels. (N x 3)
        :param voxelTSDF: TSDF value of each voxel. (N x 1)
        :param voxelWeight: Weight of each voxel. (N x 1)
        :parma truncation: Truncation value of TSDF.
        :param chunkSize: Size of chunk.
        """

        numChunk = len(voxelTSDF)
        voxelPoints = voxelPoints.reshape((-1, 3))
        voxelTSDF = voxelTSDF.reshape(-1)
        voxelWeight = voxelWeight.reshape(-1)

        voxelPoints = transformPoints(voxelPoints, frame.cameraPose.inverse())
        voxelPoints_z = voxelPoints[:, 2]

        voxelPixels = pointToPixel(voxelPoints, frame.intrinsics)
        voxelPixels_u = voxelPixels[:, 0]
        voxelPixels_v = voxelPixels[:, 1]
        
        # Filter out voxels points that visible in current frame
        imgHeight, imgWidth = frame.depth.shape
        pixelMask = torch.logical_and(voxelPixels_u >= 0,
                    torch.logical_and(voxelPixels_u < imgWidth,
                    torch.logical_and(voxelPixels_v >= 0,
                    torch.logical_and(voxelPixels_v < imgHeight,
                    voxelPoints_z > 0))))

        depthValue = torch.zeros(len(voxelPixels), dtype=torch.float32, device=self.device)
        valid_v = voxelPixels_v[pixelMask].long()
        valid_u = voxelPixels_u[pixelMask].long()
        depthValue[pixelMask] = frame.depth[valid_v, valid_u]

        # Compute truncated SDF value
        SDFValue = depthValue - voxelPoints_z
        voxelMask = torch.logical_and(depthValue > 0, torch.abs(SDFValue) < truncation)        
        TSDFValue = SDFValue[voxelMask] / truncation

        oldTSDF = voxelTSDF[voxelMask]
        oldWeight = voxelWeight[voxelMask]

        # Integrate
        newWeight = oldWeight + torch.ones_like(oldWeight, dtype=torch.float32, device=self.device)
        voxelWeight[voxelMask] = newWeight
        voxelTSDF[voxelMask] = (oldWeight * oldTSDF + TSDFValue) / newWeight

        voxelPoints = voxelPoints.reshape((numChunk, chunkSize**3, 3))
        voxelTSDF = voxelTSDF.reshape((numChunk, chunkSize, chunkSize, chunkSize))
        voxelWeight = voxelWeight.reshape((numChunk, chunkSize, chunkSize, chunkSize))

        return voxelTSDF, voxelWeight