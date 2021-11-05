import matplotlib.cm
import open3d as o3d
import numpy as np

class BBox:
    def __init__(self, min, max) -> None:
        self.min = min
        self.max = max

    def getBboxGeometry(self, color=0):
        points = np.asarray([
            [self.min[0], self.min[1], self.min[2]],
            [self.min[0], self.min[1], self.max[2]],
            [self.min[0], self.max[1], self.min[2]],
            [self.min[0], self.max[1], self.max[2]],
            [self.max[0], self.min[1], self.min[2]],
            [self.max[0], self.min[1], self.max[2]],
            [self.max[0], self.max[1], self.min[2]],
            [self.max[0], self.max[1], self.max[2]]
        ])

        lineIndices = np.asarray([
            [0, 1], [2, 3], [4, 5], [6, 7],
            [0, 4], [1, 5], [2, 6], [3, 7],
            [0, 2], [4, 6], [1, 3], [5, 7]
        ])

        geom = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lineIndices)
        )

        my_color = np.asarray(matplotlib.cm.get_cmap('tab10').colors)[color, :3]
        geom.colors = o3d.utility.Vector3dVector(np.repeat(np.expand_dims(my_color, 0), lineIndices.shape[0], 0))

        return geom