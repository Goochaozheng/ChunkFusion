from network.utils import toSparseInput
from network.fuser import Fuser
import open3d as o3d
import torch
import spconv
import numpy as np
from time import time

from .chunkManager import ChunkManager
from geometry import transformPoints, pointToPixel
from network import Fuser, Parser


class FusionIntegrator(ChunkManager):
    """
    Reconstruction engine using the fusion network.
    """

    def __init__(self, chunkSize:int, voxelResolution:float, truncation:float, minPointsPerChunk:int, meshing:bool, device, 
        withFuser=True, withParser=True, parserModel=None, fuserModel=None,
        batchSize=128, padding=0, withBestScan=False, verbose=False) -> None:

        ChunkManager.__init__(self, chunkSize, voxelResolution, truncation, minPointsPerChunk, padding, meshing, device)

        self.verbose = verbose
        self.batchSize = batchSize
        self.withBestScan = withBestScan
        self.withFuser = withFuser
        self.withParser = withParser

        if self.withFuser:
            self.fuser = Fuser()
            self.fuser.load_state_dict(torch.load(fuserModel))
            self.fuser.eval().to(device)

        if self.withParser:
            self.parser = Parser()
            self.parser.load_state_dict(torch.load(parserModel))
            self.parser.eval().to(device)


    def integrateFrame(self, depthPointCloud):
        """
        Integrate frame using fusion network.
        """

        with torch.no_grad():

            if self.verbose:
                intersect_begin = time()

            chunkList, pointCount = self.getChunkFromPointCloud(depthPointCloud)
            if len(chunkList) == 0:
                print("- No Chunk Intersecting.")
                return

            # if self.verbose:
            #     print("- Get {} chunk intersecting. Timing: {:.06f}".format(len(chunkList), time() - intersect_begin))    
            #     allocation_begin = time()

            # Allocate chunks 
            chunkData = self.getChunkListData(chunkList, withPad=(self.padding != 0))
            voxelPoints = chunkData["voxelPoints"]
            localTSDF = chunkData["voxelTSDF"]
            localWeight = chunkData["voxelWeight"]
            obsCount = chunkData["obsCount"]

            # if self.verbose:
            #     print("- Querying {} chunks. Timing: {:.06f}".format(len(chunkList), time() - allocation_begin))    
            #     integrate_begin = time()

            if self.withBestScan:
                # For Debugging, Selective Update, Only use the scan with most points to infer the surface
                updateMask = pointCount > obsCount
                chunkList = chunkList[updateMask.cpu().numpy()]
                pointCount = pointCount[updateMask]
                voxelPoints = voxelPoints[updateMask]
                localTSDF = localTSDF[updateMask]
                localWeight = localWeight[updateMask]

            # Get Chunk TSDF
            inputTSDF = self.computeTSDF(depthPointCloud, voxelPoints, withPad=(self.padding != 0))

            inputTSDF = inputTSDF.unsqueeze(1)
            localTSDF = localTSDF.unsqueeze(1)
            localWeight = localWeight.unsqueeze(1)
        
            # Split input chunks as multiple batch
            numChunk = len(inputTSDF)
            numBatches = np.ceil(numChunk / self.batchSize)
            
            outputTSDF = torch.empty((0, self.chunkSize, self.chunkSize, self.chunkSize), device=self.device)
            outputWeight = torch.empty((0, self.chunkSize, self.chunkSize, self.chunkSize), device=self.device)

            for batchCount in np.arange(numBatches):

                dataBegin = int(batchCount * self.batchSize)
                dataEnd = int(min((batchCount + 1) * self.batchSize, numChunk))
                
                localTSDF_batch = localTSDF[dataBegin : dataEnd]
                localWeight_batch = localWeight[dataBegin : dataEnd]
                inputTSDF_batch = inputTSDF[dataBegin : dataEnd]

                inputMask = torch.abs(inputTSDF_batch) < 1
                localMask = torch.abs(localTSDF_batch) < 1
                updateMask = torch.logical_or(inputMask, localMask)

                # Predict
                if self.withFuser:
                    # Fuse TSDF using fuserNet
                    fuseTSDF_batch = self.fuser(inputTSDF_batch, localTSDF_batch)
                    fuseWeight_batch = torch.ones_like(inputTSDF_batch)
                else:
                    # Fuse TSDF with standard constant weight
                    inputWeight_batch = torch.ones_like(inputTSDF_batch)
                    fuseTSDF_batch, fuseWeight_batch = self.weightUpdate(inputTSDF_batch, inputWeight_batch, localTSDF_batch, localWeight_batch)

                if self.withParser:
                    # Refine the TSDF with parser
                    fuseTSDF_batch = self.parser(fuseTSDF_batch)
                elif self.withFuser:
                    fuseTSDF_batch = spconv.ToDense()(fuseTSDF_batch) * 2 - 1

                # Set all invalid voxel to 1 for better numerical stability
                fuseTSDF_batch[torch.abs(fuseTSDF_batch) >= 1] = 1

                # Remove Padding
                fuseTSDF_batch = fuseTSDF_batch.squeeze(1)
                fuseWeight_batch = fuseWeight_batch.squeeze(1)
                if self.padding != 0:
                    fuseTSDF_batch = fuseTSDF_batch[:,
                        self.padding : self.padding + self.chunkSize,
                        self.padding : self.padding + self.chunkSize,
                        self.padding : self.padding + self.chunkSize
                    ]

                    fuseWeight_batch = fuseWeight_batch[:,
                        self.padding : self.padding + self.chunkSize,
                        self.padding : self.padding + self.chunkSize,
                        self.padding : self.padding + self.chunkSize
                    ]

                outputTSDF = torch.cat((outputTSDF, fuseTSDF_batch), dim=0)
                outputWeight = torch.cat((outputWeight, fuseWeight_batch), dim=0)

            # Update TSDF value into chunk manager
            self.setChunkListData(chunkList, outputTSDF, outputWeight, pointCount)

            if self.verbose:
                print("- Integrated {} chunks. Timing: {:.06f}".format(len(chunkList), time() - intersect_begin))    
                # print("- Total time: {:.06f}".format(time() - intersect_begin))   

        return chunkList


    def weightUpdate(self, inputTSDF, inputWeight, localTSDF, localWeight):
        inputMask = torch.abs(inputTSDF) < 1

        validInputWeight = inputWeight[inputMask]
        validLocalWeight = localWeight[inputMask]
        validNewWeight = validInputWeight + validLocalWeight

        localTSDF[inputMask] = (inputTSDF[inputMask] * validInputWeight + localTSDF[inputMask] * validLocalWeight) / validNewWeight
        localWeight[inputMask] = validNewWeight

        return localTSDF, localWeight


    def computeTSDF(self, frame, voxelPoints, withPad=True):
        """
        Compute the TSDF value of given point cloud.

        :param frame: DepthPointCloud of current frame, containing depth map, intrinsics & pose.
        :param voxelPoints: 3D coordinates of all voxels. (N x 4096 x 3)
        ;param withPad: padding chunk or not
        :return TSDFValue: The TSDF value computed from input frame. (N x 4096)
        :return inputWeight: The observation mark of voxel. (N x 4096)
        """

        numChunks = len(voxelPoints)
        voxelPoints = voxelPoints.reshape((-1, 3))
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
        voxelMask = torch.logical_and(depthValue > 0, torch.abs(SDFValue) < self.truncation)        
        TSDFValue = torch.ones_like(SDFValue)
        TSDFValue[voxelMask] = SDFValue[voxelMask] / self.truncation

        if withPad:
            TSDFValue = TSDFValue.reshape((numChunks, self.paddedSize, self.paddedSize, self.paddedSize))
        else:
            TSDFValue = TSDFValue.reshape((numChunks, self.chunkSize, self.chunkSize, self.chunkSize))

        return TSDFValue



    def tsdfUpdate(self, frame, voxelPoints, voxelTSDF, voxelWeight):
        """
        TSDF Updating.

        :param frame: DepthPointCloud of current frame, containing depth map, intrinsics & pose.
        :param voxelPoints: 3D coordinates of all voxels. (N x 3)
        :param voxelTSDF: TSDF value of each voxel. (N x 1)
        :param voxelWeight: Weight of each voxel. (N x 1)
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
        voxelMask = torch.logical_and(depthValue > 0, torch.abs(SDFValue) < self.truncation)        
        TSDFValue = SDFValue[voxelMask] / self.truncation

        oldTSDF = voxelTSDF[voxelMask]
        oldWeight = voxelWeight[voxelMask]

        # Integrate
        newWeight = oldWeight + torch.ones_like(oldWeight, dtype=torch.float32, device=self.device)
        voxelWeight[voxelMask] = newWeight
        voxelTSDF[voxelMask] = (oldWeight * oldTSDF + TSDFValue) / newWeight

        voxelPoints = voxelPoints.reshape((numChunk, self.paddedSize**3, 3))
        voxelTSDF = voxelTSDF.reshape((numChunk, self.paddedSize, self.paddedSize, self.paddedSize))
        voxelWeight = voxelWeight.reshape((numChunk, self.paddedSize, self.paddedSize, self.paddedSize))

        return voxelTSDF, voxelWeight



    def inverseTSDF(self, inputTSDF:torch.Tensor):
        """
        Convert the TSDF into inverse TSDF.
        """
        outputTSDF = torch.zeros_like(inputTSDF) 
        outputTSDF[inputTSDF == 0] = 1
        outputTSDF[inputTSDF > 0] = 1 - inputTSDF[inputTSDF > 0]
        outputTSDF[inputTSDF < 0] = -1 - inputTSDF[inputTSDF < 0]
        return outputTSDF