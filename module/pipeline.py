import os
import numpy as np
import open3d as o3d
import torch
import cv2
from glob import glob
from os.path import join
from scipy.spatial.transform import Rotation

from .chunkManager import constructChunksFromVolume
from .fusionIntegrator import FusionIntegrator
from .tsdfIntegrator import TSDFIntegrator
from geometry import Camera
from geometry import DepthPointCloud
from geometry import BBox

class Pipeline:
    """
    Pipeline frame sequence integration.
    """

    def __init__(self, config) -> None:

        self.config = config
        self.verbose = config.verbose

        # Set up GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(0)
        # self.device = "cpu"

        # Chunk Manager for integration
        self.integrator = None

        # Chunk Manager for ground truth voxel management
        self.chunkManger_gt = None

        # Camera Parameter
        self.intrinsics = None
        self.poseList = None
        self.frameList = None
        self.frameCount = None
        self.depthCut = config.depthCut

        # Parameter for visualization control
        self.index = 0
        self.nextStep = False
        self.play = False
        self.showGroundTruth = self.config.groundTruth and self.config.dataset == "ModelNet"
        self.meshDisplay = "fusion"
        self.chunkUpdate = None
        self.freeView = config.freeView
        self.showDepth = config.showDepth
        self.cameraOffset = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ])

        # Visualization geometries management
        self.geometryMap = {}
        self.chunkVisMap = {}
        self.chunkMeshMap = {}
        self.gtMeshMap = {}


    
    def run(self):
        """
        Start the pipeline.
        Initialize chunk manager and start visualizer.
        """

        print(
        "-------------------------------------------------------\n",
        "Starting ChunkFusion pipeline.\n\n",
        "- Press \"SPACE\" to process next frame;\n",
        "- Press \".\" to continuously process frames; Press again to stop;\n",
        "- Press \"S\" to save current mesh;\n",
        "- Press \"F\" to toggle free viewport mode;\n"
        )

        self.intrinsics = np.array([
            [self.config.fx, 0., self.config.cx],
            [0., self.config.fy, self.config.cy],
            [0., 0., 1.]
        ])

        # Read camera pose list
        if self.config.dataset == "ModelNet":
            self.poseList = np.sort(glob(join(self.config.dataRoot, "scan", "*.pose.txt")))
        elif self.config.dataset == "ICL-NUIM":
            poseFile = glob(join(self.config.dataRoot, "*.gt.freiburg"))[0]
            self.poseList = np.loadtxt(poseFile)[:, 1:]
        elif self.config.dataset == "orbbec":
            poseFile = join(self.config.dataRoot, "KeyFrameTrajectory.txt")
            self.poseList = np.loadtxt(poseFile, dtype=str)
            self.timestampList = self.poseList[:, 0]
            self.poseList = self.poseList[:, 1:].astype(float)
        elif self.config.dataset == "d435i":
            poseFile = join(self.config.dataRoot, "poses.txt")
            self.poseList = np.loadtxt(poseFile)[:, 1:]            
        elif self.config.dataset == "Demo":
            self.poseList = np.sort(glob(join(self.config.dataRoot, "frame-*.pose.txt")))
        else:
            raise Exception("Unrecognized Dataset.")


        # Read depth frame list
        if self.config.dataset == "ModelNet":
            if self.config.noiseScan:
                self.frameList = np.sort(glob(join(self.config.dataRoot, "scan", "*.depth.noise.png")))
            else:
                self.frameList = np.sort(glob(join(self.config.dataRoot, "scan", "*.depth.png")))
        elif self.config.dataset == "ICL-NUIM":
            frameCount = len(glob(join(self.config.dataRoot, "depth", "*.png")))
            self.frameList = [ join(self.config.dataRoot, "depth", "{:d}.png".format(i)) for i in range(1, frameCount) ]
        elif self.config.dataset == "orbbec":
            self.frameList = [
                join(self.config.dataRoot, "depth", "{}.png".format(timestamp))
                for timestamp in self.timestampList
            ]
        elif self.config.dataset == "d435i":            
            self.frameList = np.sort(glob(join(self.config.dataRoot, "depth", "*.png")))            
        elif self.config.dataset == "Demo":
            self.frameList = np.sort(glob(join(self.config.dataRoot, "frame-*.depth.png")))
        else:
            raise Exception("Unrecognized Dataset.")

        assert len(self.poseList) == len(self.frameList)
        self.frameCount = len(self.frameList)

        # Initialize chunk manager for voxel management.
        if self.config.mode == "TSDF":
            self.integrator = TSDFIntegrator(
                chunkSize=self.config.chunkSize, 
                voxelResolution=self.config.voxelResolution, 
                truncation=self.config.truncation, 
                minPointsPerChunk=self.config.minPointsPerChunk,
                meshing=self.config.meshing,
                device=self.device,
                verbose=self.config.verbose
            )
        elif self.config.mode == "ChunkFusion":
            self.integrator = FusionIntegrator(
                chunkSize=self.config.chunkSize, 
                voxelResolution=self.config.voxelResolution, 
                truncation=self.config.truncation, 
                minPointsPerChunk=self.config.minPointsPerChunk,
                
                meshing=self.config.meshing,
                device=self.device,
                withFuser=self.config.withFuser,
                withParser=self.config.withParser,
                parserModel=self.config.parserModel,
                fuserModel=self.config.fuserModel,
                batchSize=self.config.batchSize,
                padding=self.config.padding,
                withBestScan=self.config.withBestScan,
                verbose=self.config.verbose,
            )

        if self.verbose:
            print(
            "- Chunk Size: {}\n".format(self.config.chunkSize),
            "- Voxel Size: {:.5f}\n".format(self.config.voxelResolution),
            "- Truncation Margin: {:.5f}\n".format(self.config.truncation),
            "-------------------------------------------------------\n"
            )

        # Set up ground truth chunk manager
        if self.showGroundTruth:
            voxel_gt = np.load(join(self.config.dataRoot, "voxel_sdf.npy"))
            self.chunkManger_gt = constructChunksFromVolume(
                torch.from_numpy(voxel_gt).to(self.device), 
                chunkSize=self.config.chunkSize, 
                voxelResolution=self.config.voxelResolution, 
                truncation=self.config.truncation, 
                minPointsPerChunk=self.config.minPointsPerChunk,
                meshing=self.config.meshing,
                device=self.device
            )        

        # Set up open3d visualization
        window = o3d.visualization.VisualizerWithKeyCallback()
        window.create_window(window_name="ChunkFusion", width=1280, height=720, visible=True)
        window.get_render_option().mesh_show_back_face = True
        window.register_key_callback(key=ord(" "), callback_func=self.step)
        window.register_key_callback(key=ord("."), callback_func=self.continuous)
        window.register_key_callback(key=ord("S"), callback_func=self.saveMesh)
        window.register_key_callback(key=ord("F"), callback_func=self.toggleFreeView)
        # For Debugging
        # window.register_key_callback(key=ord("C"), callback_func=self.switchMesh)

        window.register_animation_callback(callback_func=self.nextFrame)
        window.get_render_option().line_width = 50

        origin = o3d.geometry.TriangleMesh.create_coordinate_frame()
        origin.scale(0.5, center=origin.get_center())
        window.add_geometry(origin)

        bbox = BBox([-3, -3, -3], [3, 3, 3]).getBboxGeometry()
        window.add_geometry(bbox)
        window.remove_geometry(bbox, reset_bounding_box=False)

        cameraVis = Camera(scale=0.1, pose=np.eye(4)).getCameraGeometry()
        self.updateGeometry(window, "camera", cameraVis)
        
        window.run()
        window.destroy_window()


    def nextFrame(self, vis):
        """
        Visualization loop callback.
        Process next frame.
        """

        if not self.play and not self.nextStep:
            return False

        if self.index >= self.frameCount:
            self.play = False
            self.nextStep = False
            print("- End of sequence.")
            return False

        print("- Processing Frame: {}".format(self.index))

        # Read depth frame
        depthFileName = self.frameList[self.index]
        if not os.path.isfile(depthFileName):
            print("# File not found: {}".format(depthFileName))
            self.play = False
            self.nextStep = False
            return 
        depthImage = cv2.imread(depthFileName, cv2.IMREAD_ANYDEPTH).astype(float)
        depthImage /= self.config.depthScale

        # Truncate depth value
        if self.depthCut > 0:
            depthImage[depthImage > self.depthCut] = 0

        # Show depth image
        if self.showDepth:
            imgDisplay = cv2.convertScaleAbs(depthImage, alpha=255.0/10.0)
            imgDisplay = cv2.applyColorMap(imgDisplay, colormap=cv2.COLORMAP_JET)
            imgDisplay[depthImage == 0] = np.array([0,0,0])
            cv2.imshow("Depth", imgDisplay)
            cv2.waitKey(1)

        # Read pose 
        if self.config.dataset == "ICL-NUIM":
            pose = self.poseList[self.index]
            translation = pose[:3]
            translation[1] *= -1
            quat = pose[3:]
            rotation = Rotation.from_quat(quat).as_matrix()
            rotation[1] *= -1
            rotation[:, 1] *= -1
            cameraPose = np.eye(4)
            cameraPose[:3, :3] = rotation
            cameraPose[:3, 3] = translation
        elif self.config.dataset == "orbbec" or self.config.dataset == "d435i":
            pose = self.poseList[self.index]
            translation = pose[:3]
            quat = pose[3:]
            rotation = Rotation.from_quat(quat).as_matrix()
            cameraPose = np.eye(4)
            cameraPose[:3, :3] = rotation
            cameraPose[:3, 3] = translation
        else:
            poseFilename = self.poseList[self.index]
            if not os.path.isfile(poseFilename):
                print("# File not found: {}".format(poseFilename))
                self.play = False
                self.nextStep = False
                return 
            cameraPose = np.loadtxt(poseFilename)

        # Construct Point Cloud
        depthPointCloud = DepthPointCloud(
            torch.from_numpy(depthImage).to(self.device), 
            torch.from_numpy(self.intrinsics).to(self.device),
            torch.from_numpy(cameraPose).to(self.device)
        )

        camera = Camera(scale=0.1, pose=cameraPose)
        # frustum
        self.updateGeometry(vis, "camera", camera.getCameraGeometry())

        if not self.freeView:
            view_ctl = vis.get_view_control()
            cam = view_ctl.convert_to_pinhole_camera_parameters()
            cam.extrinsic = np.matmul(self.cameraOffset, np.linalg.inv(cameraPose))
            view_ctl.convert_from_pinhole_camera_parameters(cam)


        if self.config.showPointcloud:
            self.updateGeometry(vis, "pointCloud", depthPointCloud.getPointCloudGeometry())

        if self.index % self.config.integrateInterval == 0:
            
            # Integrate Frame
            # TODO: Put this in a separated thread

            chunkUpdate = self.integrator.integrateFrame(depthPointCloud)

            # Recompute mesh
            if self.config.meshing:
                self.integrator.updateMesh(chunkUpdate)
                if self.showGroundTruth:
                    self.chunkManger_gt.updateMesh(chunkUpdate)

            # Compute error between ground-truth 
            if self.showGroundTruth:
                outputTSDF = self.integrator.getChunkListData(chunkUpdate, withPad=False)["voxelTSDF"]
                groundTruthTSDF = self.chunkManger_gt.getChunkListData(chunkUpdate, withPad=False)["voxelTSDF"]
                mask = torch.abs(outputTSDF) < 1
                error = torch.nn.MSELoss(reduction="mean")(outputTSDF[mask], groundTruthTSDF[mask]).item()
                print("- MSE Error: {}".format(error))

            if chunkUpdate is not None:

                # Update Chunk visual
                self.inactiveChunkVis(vis, mode="remove")

                for chunkID in chunkUpdate:
                    chunk = self.integrator.getChunk(chunkID)
                    chunkBox = chunk.getChunkGeometry(color=2)
                    self.updateChunkVis(vis, chunkID, chunkBox)

                    if self.config.meshing:
                        chunkMesh = self.integrator.getMesh(chunkID)
                        self.updateChunkMesh(vis, chunkID, chunkMesh)
                        
                        if self.showGroundTruth:
                            gtMesh = self.chunkManger_gt.getMesh(chunkID)
                            self.updateGTMesh(vis, chunkID, gtMesh)

                chunkUpdate = None


        self.index += 1
        self.nextStep = False


    """
    Methods for visualization management.
    """
    def step(self, vis):
        self.nextStep = not self.nextStep


    def continuous(self, vis):
        self.play = not self.play

    # For Debugging
    # def switchMesh(self, vis):
    #     if self.meshDisplay == "fusion":
    #         for chunkID in self.chunkMeshMap.keys():
    #             vis.remove_geometry(self.chunkMeshMap[chunkID], reset_bounding_box=False)
    #         for chunkID in self.gtMeshMap.keys():
    #             vis.add_geometry(self.gtMeshMap[chunkID], reset_bounding_box=False)
    #         self.meshDisplay = "gt"
    #         print("- Switching To Ground Truth")


    #     elif self.meshDisplay == "gt":
    #         for chunkID in self.gtMeshMap.keys():
    #             vis.remove_geometry(self.gtMeshMap[chunkID], reset_bounding_box=False)
    #         for chunkID in self.chunkMeshMap.keys():
    #             vis.add_geometry(self.chunkMeshMap[chunkID], reset_bounding_box=False)
    #         self.meshDisplay = "fusion"      
    #         print("Switching To Fusion")



    def updateGeometry(self, vis, ID, geometry):
        if ID in self.geometryMap.keys():
            vis.remove_geometry(self.geometryMap[ID], reset_bounding_box=False)

        self.geometryMap[ID] = geometry
        vis.add_geometry(geometry, reset_bounding_box=False)


    def updateChunkVis(self, vis, chunkID, chunkGeometry):
        chunkID = tuple(chunkID)
        if chunkID in self.chunkVisMap.keys():
            vis.remove_geometry(self.chunkVisMap[chunkID], reset_bounding_box=False)

        self.chunkVisMap[chunkID] = chunkGeometry
        vis.add_geometry(chunkGeometry, reset_bounding_box=False)


    def updateChunkMesh(self, vis, chunkID, chunkMesh):
        chunkID = tuple(chunkID)
        if chunkID in self.chunkMeshMap.keys():
            vis.remove_geometry(self.chunkMeshMap[chunkID], reset_bounding_box=False)

        self.chunkMeshMap[chunkID] = chunkMesh
        if self.meshDisplay == "fusion":
            vis.add_geometry(chunkMesh, reset_bounding_box=False)


    def updateGTMesh(self, vis, chunkID, chunkMesh):
        chunkID = tuple(chunkID)
        if chunkID in self.gtMeshMap.keys():
            vis.remove_geometry(self.gtMeshMap[chunkID], reset_bounding_box=False)

        self.gtMeshMap[chunkID] = chunkMesh
        if self.meshDisplay == "gt":
            vis.add_geometry(chunkMesh, reset_bounding_box=False)


    def removeChunkVis(self, vis, chunkID):
        if chunkID in self.chunkVisMap.keys():
            vis.remove_geometry(self.chunkVisMap[chunkID], reset_bounding_box=False)


    def inactiveChunkVis(self, vis, mode="remove"):
        if mode == "remove":
            for chunkID in self.chunkVisMap.keys():
                self.removeChunkVis(vis, chunkID)
        elif mode == "gray":
            for chunkID in self.chunkVisMap.keys():
                colorGray = np.array([0.9, 0.9, 0.9])
                self.chunkVisMap[chunkID].colors = o3d.utility.Vector3dVector(np.repeat(np.expand_dims(colorGray, 0), 12, 0))


    def saveMesh(self, vis):
        print("- Saving Mesh ...")
        self.integrator.saveAllMesh("./result/mesh.ply")


    def toggleFreeView(self, vis):
        self.freeView = not self.freeView
        if self.freeView:
            print("- Free Viewport On")
        else:
            print("- Free Viewport Off")

