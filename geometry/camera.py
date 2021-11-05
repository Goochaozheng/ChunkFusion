import numpy as np
import open3d as o3d
import matplotlib.cm

class Camera:
    def __init__(self, scale, pose):
        self.pose = pose
        self.scale = scale

        aspect_ratio = 4. / 3.
        pw = np.tan(np.deg2rad(90 / 2.)) * scale
        ph = pw / aspect_ratio
        self.corners = np.asarray([
            [0.0, 0.0, 0.0],
            [pw, ph, scale],
            [pw, -ph, scale],
            [-pw, ph, scale],
            [-pw, -ph, scale],
            [0.0, ph, 0.0],
            [pw, 0.0, 0.0]
        ])

        self.line_indices = np.asarray([
            [0, 1], [0, 2], [0, 3], [0, 4],
            [1, 2], [1, 3], [3, 4], [2, 4],
            [0,5 ], [0, 6]
        ])

        self.transform(pose)


    def transform(self, pose: np.ndarray):
        corners_homo = np.concatenate((self.corners, np.ones((len(self.corners), 1))), axis=1)
        corners_transformed = np.matmul(pose, corners_homo.T).T
        self.corners = corners_transformed[:, :3]


    def getCameraGeometry(self):

        geom = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(self.corners),
            lines=o3d.utility.Vector2iVector(self.line_indices))

        my_color = np.asarray(matplotlib.cm.get_cmap('tab10').colors)[3, :3]
        geom.colors = o3d.utility.Vector3dVector(np.repeat(np.expand_dims(my_color, 0), self.line_indices.shape[0], 0))

        return geom