import numpy as np
from mayavi import mlab

from pathlib import Path

root_split_path = Path("/media/wangkai/MyPassport/dataset/kitti/object_openpcd/testing")


def get_lidar(idx):
    lidar_file = root_split_path / 'velodyne' / ('%06d.bin' % idx)
    assert lidar_file.exists()
    lidar_points = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
    # 没有mvp也没有super这个时候通道是4

    points = np.concatenate([lidar_points, np.ones([lidar_points.shape[0], 4])], axis=1)

    three_types_points_file = root_split_path / 'three_types_points' / ('%06d.npy' % idx)
    three_types_points_file = np.load(three_types_points_file, allow_pickle=True).item()

    painted_points = three_types_points_file['real_points']
    if painted_points is not None:
        painted_points = np.concatenate([painted_points, np.zeros([painted_points.shape[0], 1])], axis=1)

    super_points = three_types_points_file['super_points']
    if super_points is not None:
        super_points = np.concatenate([super_points, -2 * np.ones([super_points.shape[0], 1])], axis=1)

    virtual_points = three_types_points_file['virtual_points']
    if virtual_points is not None:
        virtual_points = np.concatenate([virtual_points, -1 * np.ones([virtual_points.shape[0], 1])], axis=1)

    return points, painted_points, super_points, virtual_points


def show_point(*points_list):
    fig = mlab.figure("point", bgcolor=(0, 0, 0), size=(1650, 1500))

    for i, points in enumerate(points_list):
        x = points[:, 0]  # x position of point
        y = points[:, 1]  # y position of point
        z = points[:, 2] + i * 0.01  # z position of point
        colors = [(1, 0, 0), (0, 1, 0), (1, 1, 1)]

        mlab.points3d(x, y, z,
                      scale_factor=0.05 - 0.01 * i,
                      # z,
                      color=colors[i],  # Values used for Color
                      # mode="point",
                      mode="sphere",
                      colormap='spectral',  # 'bone', 'copper', 'gnuplot'
                      # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                      figure=fig,
                      )

    mlab.show()


if __name__ == "__main__":
    points, painted_points, super_points, virtual_points = get_lidar(10)
    print(points.shape, painted_points.shape,
          super_points.shape, virtual_points.shape)
    show_point(points, virtual_points)
