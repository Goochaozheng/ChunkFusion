import torch
import open3d as o3d
from .utils import pixelToPoint, transformPoints


class DepthPointCloud:
    def __init__(self, depthImg:torch.Tensor, intrinsics:torch.Tensor, cameraPose:torch.Tensor):
        
        self.depth = depthImg.float()
        self.intrinsics = intrinsics.float()
        self.cameraPose = cameraPose.float()

        self.points = pixelToPoint(self.depth, self.intrinsics)
        self.points = transformPoints(self.points, self.cameraPose)

    def getPointCloudGeometry(self):
        points = self.points.cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd
