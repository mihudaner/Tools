import numpy as np
import mayavi.mlab



def show_point(real_pointcloud1, real_pointcloud2):
    x1 = real_pointcloud1[:, 0]  # x position of point
    y1 = real_pointcloud1[:, 1]  # y position of point
    z1 = real_pointcloud1[:, 2]  # z position of point

    x3 = real_pointcloud1[326*700+450, 0]  # x position of point
    y3 = real_pointcloud1[326*700+450, 1]  # y position of point
    z3 = real_pointcloud1[326*700+450, 2]  # z position of point

    x2 = real_pointcloud2[:, 0]  # x position of point
    y2 = real_pointcloud2[:, 1]  # y position of point
    z2 = real_pointcloud2[:, 2]  # z position of point

    x4 = real_pointcloud2[342*700+460, 0]  # x position of point
    y4 = real_pointcloud2[342*700+460, 1]  # y position of point
    z4 = real_pointcloud2[342*700+460, 2]  # z position of point

    fig = mayavi.mlab.figure("point", bgcolor=(0, 0, 0), size=(650, 500))

    mayavi.mlab.points3d(x1, y1, z1,
                         scale_factor=.05,
                         # z,
                         color=(1, 0, 0),  # Values used for Color
                         mode="point",
                         colormap='spectral',  # 'bone', 'copper', 'gnuplot'
                         # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                         figure=fig,
                         )

    mayavi.mlab.points3d(x2, y2, z2,
                         scale_factor=.05,
                         # z,
                         color=(0, 1, 0),  # Values used for Color
                         mode="point",
                         colormap='spectral',  # 'bone', 'copper', 'gnuplot'
                         # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                         figure=fig,
                         )

    mayavi.mlab.points3d(x3, y3, z3,
                         scale_factor=.2,
                         # z,
                         color=(0, 1, 0),  # Values used for Color
                         mode="sphere",
                         colormap='spectral',  # 'bone', 'copper', 'gnuplot'
                         # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                         figure=fig,
                         )

    mayavi.mlab.points3d(x4, y4, z4,
                         scale_factor=.2,
                         # z,
                         color=(0, 1, 0),  # Values used for Color
                         mode="sphere",
                         colormap='spectral',  # 'bone', 'copper', 'gnuplot'
                         # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                         figure=fig,
                         )

    mayavi.mlab.show()


if __name__ == '__main__':
    real_pointcloud1 = np.fromfile(r"C:\Users\33567\Desktop\match\plys\tools/000279.bin", dtype=np.float64, count=-1).reshape([-1, 4])
    real_pointcloud2 = np.fromfile(r"C:\Users\33567\Desktop\match\plys\tools/res.bin", dtype=np.float64, count=-1).reshape([-1, 4])

    show_point(real_pointcloud1, real_pointcloud2)
