import matplotlib.cm
import numpy as np
import open3d as o3d

class Frustum:
    
    def __init__(self, intrinsics, imgWidth, imgHeight, nearRange: float = 0.5, farRange: float = 5., pose: np.ndarray = np.eye(4)) -> None:

        self.intrinsics = intrinsics
        self.nearRange = nearRange
        self.farRange = farRange
        self.imgWidth = imgWidth
        self.imgHeight = imgHeight

        self.corners = []
        self.lineIndices = np.array([
            [0, 1], [3, 2], [1, 3], [2, 0],
            [4, 7], [6, 5], [5, 7], [6, 4],
            [0, 5], [1, 6], [2, 7], [3, 4]
        ])

        fx = intrinsics[0,0]
        fy = intrinsics[1,1]
        cy = intrinsics[1,2]

        aspect = (fx * imgWidth) / (fy * imgHeight)
        fov = np.arctan2(cy, fy) + np.arctan2(imgHeight - cy, fy)
        tan = np.tan(fov / 2)

        heightNear = tan * nearRange
        widthNear = heightNear * aspect

        heightFar = tan * farRange
        widthFar = heightFar * aspect

        self.corners = np.array([
            [-widthFar, -heightFar, farRange],      # far-top-left
            [widthFar, -heightFar, farRange],       # far-top-right
            [-widthFar, heightFar, farRange],       # far-bot-left
            [widthFar, heightFar, farRange],        # far-bot-right
            [widthNear, heightNear, nearRange],     # near-bot-right
            [-widthNear, -heightNear, nearRange],   # near-top-left
            [widthNear, -heightNear, nearRange],    # near-top-right
            [-widthNear, heightNear, nearRange],    # near-bot-left
        ])

        corners_homo = np.concatenate((self.corners, np.ones((8, 1))), axis=1)
        corners_transformed = np.matmul(pose, corners_homo.T).T
        self.corners = corners_transformed[:, :3]


    def transform(self, pose: np.ndarray):
        """
        Transform the frustum to current camera position.
        """
        corners_homo = np.concatenate((self.corners, np.ones((8, 1))), axis=1)
        corners_transformed = np.matmul(pose, corners_homo.T).T
        self.corners = corners_transformed[:, :3]


    def getFrsutumGeometry(self, color=0):

        geom = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(self.corners),
            lines=o3d.utility.Vector2iVector(self.lineIndices)
        )

        my_color = np.asarray(matplotlib.cm.get_cmap('tab10').colors)[color, :3]
        geom.colors = o3d.utility.Vector3dVector(np.repeat(np.expand_dims(my_color, 0), self.lineIndices.shape[0], 0))

        return geom


