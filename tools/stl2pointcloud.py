"""将stl文件转为ply"""

import trimesh
import open3d as o3d


def stl_to_point_cloud(stl_path, sample_points=10000):
    # 验证输入参数
    if not isinstance(stl_path, str) or not isinstance(sample_points, int) or sample_points <= 0:
        raise ValueError("Invalid input parameters")

    try:
        # 加载 STL 文件
        mesh = trimesh.load_mesh(stl_path)
        # 采样点云
        points = mesh.sample(sample_points)
        return points
    except Exception as e:
        print(f"Error loading or sampling the mesh: {e}")
        return None


def save_point_cloud_as_ply(points, filename):
    # 验证输入参数
    if points is None or not isinstance(filename, str):
        raise ValueError("Invalid input parameters")

    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(filename, pcd, write_ascii=True)
    except Exception as e:
        print(f"Error saving point cloud: {e}")


if __name__ == '__main__':
    stl_path = '/home/yhlever/DeepLearning/6D_object_pose_estimation/Datasets/HANDLE/models/handle.stl'

    # 生成点云
    points = stl_to_point_cloud(stl_path, sample_points=10000)
    if points is None:
        print("Failed to generate point cloud")
    else:
        # 点云文件保存路径
        point_cloud_file = '/home/yhlever/DeepLearning/6D_object_pose_estimation/Datasets/HANDLE/models/handle_point_cloud.ply'

        # 保存点云数据
        save_point_cloud_as_ply(points, point_cloud_file)