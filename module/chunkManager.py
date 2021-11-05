import torch
import numpy as np
import open3d as o3d
from skimage.measure import marching_cubes

from .chunk import Chunk
from geometry import DepthPointCloud

class ChunkManager:

    def __init__(self, chunkSize:int, voxelResolution:float, truncation:float, minPointsPerChunk:int, padding:int, meshing:bool, device):
        """
        ChunkManager, managing chunked voxels.

        :param chunkSize: int, Size of the voxel chunk.
        :param voxelResolution: float, Resolution of the voxel
        :param truncation: float, Truncation margin of TSDF.
        """

        self.chunkSize = chunkSize
        self.chunkShape = (chunkSize, chunkSize, chunkSize)
        self.voxelResolution = voxelResolution
        self.padding = padding
        self.paddedSize = chunkSize + 2 * padding

        self.meshing = meshing
        self.truncation = truncation
        self.minPointsPerChunk = minPointsPerChunk
        self.device = device

        # Voxel coordinates
        x, y, z = torch.meshgrid(
            torch.arange(0, self.chunkSize, 1),
            torch.arange(0, self.chunkSize, 1),
            torch.arange(0, self.chunkSize, 1),
        )        

        self.initVoxelCoords = torch.cat([
            x.reshape(1, -1),
            y.reshape(1, -1),
            z.reshape(1, -1)
        ], dim=0).int().T + 0.5
        self.initVoxelCoords = self.initVoxelCoords.to(self.device)
        self.initVoxelCoords.requires_grad = False

        if padding != 0:
            x, y, z = torch.meshgrid(
                torch.arange(-padding, self.chunkSize + padding, 1),
                torch.arange(-padding, self.chunkSize + padding, 1),
                torch.arange(-padding, self.chunkSize + padding, 1),
            )
            self.paddedVoxelCoords = torch.cat([
                x.reshape(1, -1),
                y.reshape(1, -1),
                z.reshape(1, -1)
            ], dim=0).int().T + 0.5
            self.paddedVoxelCoords = self.paddedVoxelCoords.to(self.device)
            self.paddedVoxelCoords.requires_grad = False


        # Hashed Map {3D-ChunkID: Chunk}
        self._chunkMap = {}

        # Hashed Map {3D-ChunkID: Mesh}
        self._meshMap = {}

        self.paddingOffsets = np.array([
            [ 1,  0,  0],   [ 0,  1,  0],   [ 0,  0,  1], 
            [-1,  0,  0],   [ 0, -1,  0],   [ 0,  0, -1],  
            [ 1,  1,  0],   [ 1, -1,  0],   [-1,  1,  0],
            [-1, -1,  0],   [ 0,  1,  1],   [ 0,  1, -1],
            [ 0, -1,  1],   [ 0, -1, -1],   [ 1,  0,  1],
            [ 1,  0, -1],   [-1,  0,  1],   [-1,  0, -1],
            [ 1,  1,  1],   [ 1,  1, -1],   [ 1, -1,  1],
            [-1,  1,  1],   [ 1, -1, -1],   [-1,  1, -1],
            [-1, -1,  1],   [-1, -1, -1]
        ])


    def reset(self):
        del self._chunkMap, self._meshMap

        self._chunkMap = {}
        self._meshMap = {}

    ############################################################
    # Utility method for chunk management:
    # - Add / Get / Remove chunks;
    #############################################################

    def hasChunk(self, chunkID) -> bool:
        return tuple(chunkID) in self._chunkMap


    def hasMesh(self, chunkID) -> bool:
        return tuple(chunkID) in self._meshMap


    def allocateChunk(self, chunkID):
        """
        Allocate new chunk in the map.
        """
        if self.hasChunk(chunkID):
            return
        else:
            chunkID = tuple(chunkID)
            self._chunkMap[chunkID] = Chunk(chunkID, self.chunkSize, self.voxelResolution, self.device)

        if self.meshing:
            if self.hasMesh(chunkID):
                return
            else:
                self._meshMap[chunkID] = o3d.geometry.TriangleMesh()


    def getAllChunkID(self):
        return np.array(list(self._chunkMap.keys()))


    def getChunk(self, chunkID) -> Chunk:
        if self.hasChunk(chunkID):
            chunkID = tuple(chunkID)
            return self._chunkMap[chunkID]
        else:
            raise Exception("Chunk ID not found")



    def getChunkListData(self, chunkList, withPad=True):
        """
        Return the array of all target chunks data: Coords of voxels & SDF value & weights & points within the chunk.
        Not existing chunkID will be allocated.
        """

        if withPad:
            voxelPoints = torch.empty((0, self.paddedSize**3, 3), dtype=torch.float32, device=self.device)
            voxelTSDF = torch.empty((0, self.paddedSize, self.paddedSize, self.paddedSize), dtype=torch.float32, device=self.device)
            voxelWeight = torch.empty((0, self.paddedSize, self.paddedSize, self.paddedSize), dtype=torch.float32, device=self.device)
        else:
            voxelPoints = torch.empty((0, self.chunkSize**3, 3), dtype=torch.float32, device=self.device)
            voxelTSDF = torch.empty((0, self.chunkSize, self.chunkSize, self.chunkSize), dtype=torch.float32, device=self.device)
            voxelWeight = torch.empty((0, self.chunkSize, self.chunkSize, self.chunkSize), dtype=torch.float32, device=self.device)            

        obsCount = torch.zeros(len(chunkList), dtype=torch.int, device=self.device)

        for idx, chunkID in enumerate(chunkList):
            if not self.hasChunk(chunkID):
                self.allocateChunk(chunkID)

            chunk = self.getChunk(chunkID)
            chunkOrigin = torch.from_numpy(chunk.chunkOrigin).to(self.device)

            chunkTSDF = chunk.getChunkTSDF()
            chunkWeight = chunk.getChunkWeight()

            obsCount[idx] = chunk.getObsCount()

            # Pad chunk volume with neighboring voxels
            if withPad:

                chunkTSDF = torch.nn.ConstantPad3d(padding=self.padding, value=1)(chunkTSDF) 
                chunkWeight = torch.nn.ConstantPad3d(padding=self.padding, value=0)(chunkWeight) 

                # Handle axis-x-pos
                for i in range(len(self.paddingOffsets)):
                    padOffset = self.paddingOffsets[i]
                    neighborID = chunkID + padOffset
                    if self.hasChunk(neighborID):
                        neighborChunk = self.getChunk(neighborID)

                        # showMesh(chunkTSDF.cpu().numpy())
                        # showMesh(neighborChunk.tsdfValues.cpu().numpy())

                        padSlice = np.empty((3,2))
                        neighborSlice = np.empty((3,2))

                        for j in range(3):
                            if padOffset[j] == 1:
                                padSlice[j] = [self.chunkSize + self.padding, self.paddedSize]
                                neighborSlice[j] = [0, self.padding]
                            elif padOffset[j] == -1:
                                padSlice[j] = [0, self.padding]
                                neighborSlice[j] = [self.chunkSize-self.padding, self.chunkSize]
                            elif padOffset[j] == 0:
                                padSlice[j] = [self.padding, self.chunkSize + self.padding]
                                neighborSlice[j] = [0, self.chunkSize]

                        chunkTSDF[
                            int(padSlice[0,0]) : int(padSlice[0,1]),
                            int(padSlice[1,0]) : int(padSlice[1,1]),
                            int(padSlice[2,0]) : int(padSlice[2,1])
                        ] = neighborChunk.getChunkTSDF()[
                            int(neighborSlice[0,0]) : int(neighborSlice[0,1]),
                            int(neighborSlice[1,0]) : int(neighborSlice[1,1]),
                            int(neighborSlice[2,0]) : int(neighborSlice[2,1])
                        ]
            
                # showMesh(chunkTSDF_temp.cpu().numpy())
                # showMesh(chunkTSDF.cpu().numpy())
                chunkVoxels = self.paddedVoxelCoords * self.voxelResolution + chunkOrigin


            else:

                chunkVoxels = self.initVoxelCoords * self.voxelResolution + chunkOrigin


            voxelPoints = torch.cat([voxelPoints, chunkVoxels.unsqueeze(0)], dim=0)
            voxelTSDF = torch.cat([voxelTSDF, chunkTSDF.unsqueeze(0)], dim=0)
            voxelWeight = torch.cat([voxelWeight, chunkWeight.unsqueeze(0)], dim=0)

        return {
            "voxelPoints": voxelPoints, 
            "voxelTSDF": voxelTSDF, 
            "voxelWeight": voxelWeight,
            "obsCount": obsCount
        }


    def getMesh(self, chunkID):
        if self.hasMesh(chunkID):
            chunkID = tuple(chunkID)
            return self._meshMap[chunkID]
        else:
            raise Exception("Chunk ID not found")


    def setChunk(self, chunkID, chunkTSDF, chunkWeight, obsCount=None):
        """
        Set the data of single chunk.
        """

        if not self.hasChunk(chunkID):
            raise Exception("Chunk ID not found")
        
        chunk = self.getChunk(chunkID)

        if chunkTSDF.shape != chunk.chunkShape:
            raise Exception("Chunk Data Size Not Match Chunk Size, expect {}, but get {} ".format(chunk.chunkShape, chunkTSDF.shape))
        chunk.tsdfValues = chunkTSDF

        if chunkWeight.shape != chunk.chunkShape:
            raise Exception("Chunk Weight Size Not Match Chunk Size, expect {}, but get {}".format(chunk.chunkShape, chunkWeight.shape))
        chunk.weightValues = chunkWeight

        if obsCount is not None:
            chunk.obsCount = obsCount
        # Update neighbor padding 


    
    def setChunkListData(self, chunkList, newTSDF, newWeights, newObsCount=None):
        """
        Set the SDF value & weights to target chunks.
        """
        for i in range(len(chunkList)):
            id = chunkList[i]
            tsdf = newTSDF[i]
            weight = newWeights[i]
            if newObsCount is None:
                self.setChunk(id, tsdf, weight)
            else:
                obsCount = newObsCount[i]
                self.setChunk(id, tsdf, weight, obsCount)
        
    
    def removeChunk(self, chunkID):
        if not self.hasChunk(chunkID) and not self.hasMesh(chunkID):
            raise Exception("Chunk ID not found")

        if self.hasChunk(chunkID):
            chunkID = tuple(chunkID)
            self._chunkMap.pop(chunkID, None)

        if self.hasMesh(chunkID):
            self._meshMap.pop(chunkID, None)


    def getChunkVolume(self, chunkIDList):
        """
        Return a voxel volume box containing target chunks
        """

        # chunkIDList = np.array(list(chunkIDList))
        chunkID_min = np.zeros(3)
        chunkID_max = np.zeros(3)

        chunkID_min[0] = np.min(chunkIDList[:, 0])
        chunkID_min[1] = np.min(chunkIDList[:, 1])
        chunkID_min[2] = np.min(chunkIDList[:, 2])

        chunkID_max[0] = np.max(chunkIDList[:, 0])
        chunkID_max[1] = np.max(chunkIDList[:, 1])
        chunkID_max[2] = np.max(chunkIDList[:, 2])

        volume_min = chunkID_min * self.chunkSize
        volume_max = (chunkID_max + 1) * self.chunkSize

        volumeSize = volume_max - volume_min

        volume = torch.ones(tuple(volumeSize.astype(int)), device=self.device)

        for chunkID in chunkIDList:

            chunk = self.getChunk(chunkID) 
            chunkSDF = chunk.getChunkTSDF()

            # get linear index of chunkID
            chunkID = chunkID - chunkID_min

            voxelIdx_begin = (chunkID * self.chunkSize).astype(int)
            voxelIdx_end = ((chunkID+1) * self.chunkSize).astype(int)
            volume[
                voxelIdx_begin[0] : voxelIdx_end[0],
                voxelIdx_begin[1] : voxelIdx_end[1],
                voxelIdx_begin[2] : voxelIdx_end[2],
            ] = chunkSDF
            
        return volume_min * self.voxelResolution, volume


    def getChunkFromPointCloud(self, pointCloud:DepthPointCloud):
        points = pointCloud.points
        actualChunkSize = self.chunkSize * self.voxelResolution
        chunkIDList = torch.floor(points / actualChunkSize)

        chunkIDList, count = torch.unique(chunkIDList, return_counts=True, dim=0)

        if self.minPointsPerChunk > 0:
            # Filter out invalid chunks with points less then threshold
            filterMask = count > self.minPointsPerChunk
            chunkIDList = chunkIDList[filterMask]
            count = count[filterMask]

        return chunkIDList.cpu().numpy(), count



    ############################################################
    # Meshing Method:
    # - Extract mesh for single chunk;
    # - Handling the mesh inside and on the border;
    # - Collect all chunk mesh;
    # - Save mesh into file;
    #############################################################

    def updateMesh(self, chunkList):
        """
        Recompute the mesh of given chunks
        """

        if not self.meshing:
            raise Exception("Meshing Not Enabled!")

        # TODO: Optimize Meshing
        for chunkID in chunkList:

            with torch.no_grad():

                mesh = self.getMesh(chunkID)
                chunk = self.getChunk(chunkID)

                hasNeighbor_x = False
                hasNeighbor_y = False
                hasNeighbor_z = False

                voxelVolume = chunk.getChunkTSDF()

                # Handle axis-x
                neighborID = chunkID + [1, 0, 0]
                if self.hasChunk(neighborID):
                    hasNeighbor_x = True
                    neighborChunk = self.getChunk(neighborID)
                    
                    # Extend border
                    voxelVolume = torch.nn.ConstantPad3d(padding=(0, 0, 0, 0, 0, 1), value=1)(voxelVolume)  
                    voxelVolume[-1, 0:self.chunkSize, 0:self.chunkSize] = neighborChunk.getChunkTSDF()[0, :, :]

                    
                # Handle axis-y
                neighborID = chunkID + [0, 1, 0]
                if self.hasChunk(neighborID):
                    hasNeighbor_y = True
                    neighborChunk = self.getChunk(neighborID)

                    # Extend border
                    voxelVolume = torch.nn.ConstantPad3d(padding=(0, 0, 0, 1, 0, 0), value=1)(voxelVolume)
                    voxelVolume[0:self.chunkSize, -1, 0:self.chunkSize] = neighborChunk.getChunkTSDF()[:, 0, :]


                # Handle axis-z
                neighborID = chunkID + [0, 0, 1]
                if self.hasChunk(neighborID):
                    hasNeighbor_z = True
                    neighborChunk = self.getChunk(neighborID)

                    # Extend border
                    voxelVolume = torch.nn.ConstantPad3d(padding=(0, 1, 0, 0, 0, 0), value=1)(voxelVolume)
                    voxelVolume[0:self.chunkSize, 0:self.chunkSize, -1] = neighborChunk.getChunkTSDF()[:, :, 0]


                # Handle border edge
                if hasNeighbor_x and hasNeighbor_y:
                    neighborID = chunkID + [1, 1, 0]
                    if self.hasChunk(neighborID):
                        neighborChunk = self.getChunk(neighborID)
                        borderVolume_xy = neighborChunk.getChunkTSDF()[0, 0, :]
                        voxelVolume[-1, -1, 0:self.chunkSize] = borderVolume_xy

                if hasNeighbor_y and hasNeighbor_z:
                    neighborID = chunkID + [0, 1, 1]
                    if self.hasChunk(neighborID):
                        neighborChunk = self.getChunk(neighborID)
                        borderVolume_yz = neighborChunk.getChunkTSDF()[:, 0, 0]
                        voxelVolume[0:self.chunkSize, -1, -1] = borderVolume_yz

                if hasNeighbor_z and hasNeighbor_x:
                    neighborID = chunkID + [1, 0, 1]
                    if self.hasChunk(neighborID):
                        neighborChunk = self.getChunk(neighborID)
                        borderVolume_zx = neighborChunk.getChunkTSDF()[0, :, 0]
                        voxelVolume[-1, 0:self.chunkSize, -1] = borderVolume_zx

                if hasNeighbor_x and hasNeighbor_y and hasNeighbor_z:
                    neighborID = chunkID + [1, 1, 1]
                    if self.hasChunk(neighborID):
                        neighborChunk = self.getChunk(neighborID)
                        voxelVolume[-1, -1, -1] = neighborChunk.getChunkTSDF()[0, 0, 0]

                # Mesh volume
                voxelVolume = voxelVolume.cpu().numpy()
                meshingMask = np.abs(voxelVolume) < 1
                
                if np.all(voxelVolume[meshingMask] > 0) or np.all(voxelVolume[meshingMask] < 0): 
                    continue

                try:
                    voxelVolume[~meshingMask] = np.nan
                    vertices, triangles, _, _ = marching_cubes(voxelVolume, level=0, allow_degenerate=False)
                    vertices += chunkID * self.chunkSize + 0.5          # Move to chunk origin position
                    vertices *= self.voxelResolution                          # Scale from unit voxel size to actual voxel size

                    mesh.vertices = o3d.utility.Vector3dVector(vertices)
                    mesh.triangles = o3d.utility.Vector3iVector(triangles)
                    mesh.compute_vertex_normals()
                
                except RuntimeError:
                    continue
                    # print("- No surface found at ", str(chunkID))



    def getAllMesh(self):
        """
        Return the mesh of all chunks.
        """

        if not self.meshing:
            raise Exception("Meshing Not Enabled!")

        allMesh = o3d.geometry.TriangleMesh()

        allVertices = np.ndarray((0, 3))
        allTriangles = np.ndarray((0, 3))

        indicesCount = 0

        if len(self._meshMap.keys()) == 0:
            print("Mesh Empty.")

        for id in self._meshMap.keys():
            chunkMesh = self.getMesh(id)
            vertices = np.asarray(chunkMesh.vertices)
            triangles = np.asarray(chunkMesh.triangles)
            if len(vertices) != 0 and len(triangles) != 0:
                allVertices = np.concatenate((allVertices, vertices), axis=0)
                triangleIndices = triangles + indicesCount
                allTriangles = np.concatenate((allTriangles, triangleIndices), axis=0)
                indicesCount += len(vertices)

        allMesh.vertices = o3d.utility.Vector3dVector(allVertices.astype(float))
        allMesh.triangles = o3d.utility.Vector3iVector(allTriangles.astype(np.int32))
        allMesh.compute_vertex_normals()

        return allMesh


    def saveAllMesh(self, filename):
        """
        Save all mesh to ply file.
        """

        if self.meshing:
            recon_mesh = self.getAllMesh()

        else:

            # Compute the mesh from the whole volume
            allChunk = self.getAllChunkID()
            origin, allVolume = self.getChunkVolume(allChunk)
            allVolume = allVolume.cpu().numpy()
            allVolume[np.abs(allVolume) >= 1] = np.nan

            vertices, triangles, _, _ = marching_cubes(allVolume, level=0)
            vertices *= self.voxelResolution
            vertices += origin + 0.5 * self.voxelResolution

            recon_mesh = o3d.geometry.TriangleMesh()
            recon_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            recon_mesh.triangles = o3d.utility.Vector3iVector(triangles)
            # recon_mesh.compute_vertex_normals()

        # Filterout nan vertices
        vertices = np.asarray(recon_mesh.vertices)
        triangles = np.asarray(recon_mesh.triangles)

        validVerticesMask = ~np.any(np.isnan(vertices), axis=1)
        validIndices = np.arange(len(vertices))[validVerticesMask]

        indicesRemap = dict(zip(validIndices, np.arange(validVerticesMask.sum())))    
        validTrianglesMask = np.all(np.isin(triangles, validIndices), axis=1)
        
        vertices = vertices[validVerticesMask]
        triangles = triangles[validTrianglesMask]

        # remap the indices of triangles
        triangles = np.vectorize(indicesRemap.__getitem__)(triangles)
        # trianglesRemapped = triangles.copy()
        # for k in indicesRemap:
        #     trianglesRemapped[triangles == k] = indicesRemap[k]

        recon_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        recon_mesh.triangles = o3d.utility.Vector3iVector(triangles)
        recon_mesh.compute_vertex_normals()

        print("- {} vertices extracted.".format(len(vertices)))
        print("- Mesh saved to: {}.".format(filename))
        o3d.io.write_triangle_mesh(filename, recon_mesh)


def constructChunksFromVolume(voxelVolume:torch.Tensor, chunkSize, voxelResolution, truncation, minPointsPerChunk, meshing, device) -> ChunkManager:
    """
    Given a voxel volume, construct the corresponding ChunkManager.
    The voxel volume center (volume_size / 2) will be placed at coordinate origin.
    """

    # Convert to truncated SDF volume
    voxelVolume.to(device)
    voxelVolume /= truncation
    voxelVolume[torch.abs(voxelVolume) > 1] = 1.

    volumeSize = np.array(voxelVolume.shape)
    voxelOrigin = -volumeSize / 2
    
    x, y, z = np.meshgrid(
        np.arange(-volumeSize[0]/(2*chunkSize), volumeSize[0]/(2*chunkSize)),
        np.arange(-volumeSize[1]/(2*chunkSize), volumeSize[1]/(2*chunkSize)),
        np.arange(-volumeSize[2]/(2*chunkSize), volumeSize[2]/(2*chunkSize)),
    )

    chunkIDList = np.concatenate((
        x.reshape(1, -1),
        y.reshape(1, -1),
        z.reshape(1, -1)
    ), axis=0).astype(int).T

    chunkManager = ChunkManager(chunkSize, voxelResolution, truncation, minPointsPerChunk, 0, meshing, device)

    for i in range(len(chunkIDList)):

        chunkID = chunkIDList[i]

        voxelIdx_begin = (chunkID * chunkSize - voxelOrigin).astype(int)
        voxelIdx_end = ((chunkID + 1) * chunkSize - voxelOrigin).astype(int)

        chunkData = voxelVolume[
            voxelIdx_begin[0] : voxelIdx_end[0],
            voxelIdx_begin[1] : voxelIdx_end[1],
            voxelIdx_begin[2] : voxelIdx_end[2],
        ]
        chunkWeight = torch.ones(chunkSize, chunkSize, chunkSize).to(device)

        chunkManager.allocateChunk(chunkID)
        chunkManager.setChunk(chunkID, chunkData, chunkWeight)
    
    return chunkManager
