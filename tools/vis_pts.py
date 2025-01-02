import numpy as np
import cv2
import open3d as o3d


def get_point_cloud_from_depth(depth, k, bbox=None):
    cam_fx, cam_fy, cam_cx, cam_cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]

    im_h, im_w = depth.shape
    xmap = np.array([[i for i in range(im_w)] for _ in range(im_h)])
    ymap = np.array([[j for _ in range(im_w)] for j in range(im_h)])

    if bbox is not None:
        rmin, rmax, cmin, cmax = bbox
        depth = depth[rmin:rmax, cmin:cmax].astype(np.float32)
        xmap = xmap[rmin:rmax, cmin:cmax].astype(np.float32)
        ymap = ymap[rmin:rmax, cmin:cmax].astype(np.float32)

    pt2 = depth.astype(np.float32)
    pt0 = (xmap.astype(np.float32) - cam_cx) * pt2 / cam_fx
    pt1 = (ymap.astype(np.float32) - cam_cy) * pt2 / cam_fy

    cloud = np.stack([pt0, pt1, pt2]).transpose((1, 2, 0))
    return cloud


def visualize_point_cloud(pt1, pt2=None):
    # 将 NumPy 数组转换为 Open3D 的 PointCloud 对象
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pt1)
    pcd1 = pcd1.voxel_down_sample(voxel_size=0.001)
    pcd1.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0] for _ in range(pt1.shape[0])], dtype=np.float64))

    # 创建一个可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 将点云添加到可视化窗口
    vis.add_geometry(pcd1)

    if pt2 is not None:
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pt2)
        pcd2.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 1] for _ in range(pt2.shape[0])], dtype=np.float64))
        vis.add_geometry(pcd2)

    # 运行可视化
    vis.run()


if __name__ == "__main__":
    depth = cv2.imread("/home/yhlever/DeepLearning/SyndataRender/syndatasets/images/depth_ims/image_000000.png", cv2.IMREAD_GRAYSCALE)
    depth = depth / 1000.0
    camera_k = np.array([[1110.6929378784982,
                          0.0,
                          638.5971943596489, ],
                         [0.0,
                          1110.6929378784982,
                          479.578884980991, ],
                         [0.0,
                          0.0,
                          1.0]])
    pts = get_point_cloud_from_depth(depth, camera_k)
    pts = pts.reshape(-1, 3)
    visualize_point_cloud(pts)
