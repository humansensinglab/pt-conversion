# Created by Baole Fang at 11/1/23

import open3d as o3d
import numpy as np
from plyfile import PlyData

print('point cloud')
plydata = PlyData.read('original_with_densification.ply')
pcd = o3d.io.read_point_cloud('original_with_densification.ply')
# fit to unit cube
colors = np.zeros((len(pcd.points), 3))
colors[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
colors[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
colors[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd])

print('voxelization')
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.01)
o3d.visualization.draw_geometries([voxel_grid])
