#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/11 14:48
# @Author  : 我的名字
# @File    : RANSAC.py
# @Description : http://www.open3d.org/docs/release/search.html?q=&check_keywords=yes&area=default#
# https://blog.csdn.net/u013019296/article/details/109349373?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168381903716800186568519%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=168381903716800186568519&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-8-109349373-null-null.142^v87^control_2,239^v2^insert_chatgpt&utm_term=python%E7%82%B9%E4%BA%91%E5%85%A8%E5%B1%80%E9%85%8D%E5%87%86&spm=1018.2226.3001.4187

import open3d as o3d
import numpy as np
import copy
import time


def draw_registration_result(source, target, transformation):
    """
    可视化输入和配准的点云
    :param source:
    :param target:
    :param transformation:
    :return:
    """
    source_temp = copy.deepcopy(source)
    source_temp.translate((0, 0, 0), relative=True)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
    return source_temp


def preprocess_point_cloud(pcd, voxel_size):
    """
    点云预处理，包括点云的下采样和提取特征
    :param pcd: 点云
    :param voxel_size: 体素大小
    :return: 下采样点云和特征
    """
    print(":: Downsample with a voxel size %.3f." % voxel_size)

    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(voxel_size):
    """
    读取点云文件和下采样提取特征
    :param voxel_size: 体素大小
    :return: source, target, source_down, target_down, source_fpfh, target_fpfh
    """
    print(":: Load two point clouds and disturb initial pose.")
    source = o3d.io.read_point_cloud(r".\plys\bun000.ply")
    target = o3d.io.read_point_cloud(r".\plys\bun045.ply")
    trans_init = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    C = np.array([0, 0, 0], dtype=np.float64)
    source.rotate(trans_init, C)

    draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    """
    RANSAC全局配准
    :param source_down:
    :param target_down:
    :param source_fpfh:
    :param target_fpfh:
    :param voxel_size:
    :return:
    """
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    return result


def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    """
    icp局部配准，需要先对点云进行全局配准才能icp配准
    :param source:
    :param target:
    :param source_fpfh:
    :param target_fpfh:
    :param voxel_size:
    :return:
    """
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
        , o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    return result


def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    """
    快速全局配准（废弃）
    :param source_down:
    :param target_down:
    :param source_fpfh:
    :param target_fpfh:
    :param voxel_size:
    :return:
    """
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
          % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result



def save_result_bin(pcd_result, reflect, path):
    """
    pcd保存为bin
    :param pcd_result:
    :param reflect:
    :param path:
    :return:
    """
    points = np.asarray(pcd_result.points)
    data = np.concatenate([points, reflect], axis=1)
    with open(path, "wb") as f:
        data.tofile(f)


def read_input_bin(path):
    """
    bin读取为pcd
    :param path: r"E:\Work\THU\code\Zivid\temp\bin\000279.bin"
    :return:
    """
    data = np.fromfile(path, dtype=np.float64)
    # 将数据重塑为点云的形状
    # 假设点云中每个点都有3个坐标和3个颜色值
    data = data.reshape([-1, 4])
    target = o3d.geometry.PointCloud()
    target_numpy = data
    target.points = o3d.utility.Vector3dVector(target_numpy[:, 0:3])
    return target


def load_save():
    source = o3d.io.read_point_cloud(r"C:\Users\33567\Desktop\match\plys/bun000.ply")
    target = o3d.io.read_point_cloud(r"C:\Users\33567\Desktop\match\plys/bun045.ply")

    # 保存点云
    reflect = np.arange(0, np.asarray(source.points).shape[0], 1).reshape((np.asarray(source.points).shape[0], 1))
    path = r"C:\Users\33567\Desktop\match\plys\bun000.bin"
    save_result_bin(source, reflect, path)

    reflect = np.arange(0, np.asarray(target.points).shape[0], 1).reshape((np.asarray(target.points).shape[0], 1))
    path = r"C:\Users\33567\Desktop\match\plys\bun045.bin"
    save_result_bin(target, reflect, path)
    return


if __name__ == '__main__':
    voxel_size = 0.05  # means 5cm for this dataset
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size)

    # ransac
    start = time.time()
    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    print("ransac %.3f sec.\n" % (time.time() - start))
    print(result_ransac)
    draw_registration_result(source_down, target_down, result_ransac.transformation)
    result = source_down.transform(result_ransac.transformation)

    # icp
    start = time.time()
    result_icp = refine_registration(source, target, source_fpfh, target_fpfh,
                                     voxel_size)
    print("icp %.3f sec.\n" % (time.time() - start))
    print(result_icp)
    res = draw_registration_result(source, target, result_icp.transformation)
    reflect = np.arange(0, np.asarray(res.points).shape[0], 1).reshape((np.asarray(res.points).shape[0], 1))

    # 全局配准
    start = time.time()
    result_fast = execute_fast_global_registration(source_down, target_down,
                                                   source_fpfh, target_fpfh,
                                                   voxel_size)
    print("Fast global  %.3f sec.\n" % (time.time() - start))
    print(result_fast)
    draw_registration_result(source_down, target_down, result_fast.transformation)
