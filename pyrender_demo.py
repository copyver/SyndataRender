import pybullet as p
import time
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import PIL.Image as Image
import open3d as o3d
import pyrender
import trimesh
from scipy.spatial.transform import Rotation as R

use_gui = True
use_pyrender = True
# 连接到pybullet
if use_gui:
    physicsClient = p.connect(p.GUI)
    p_render = p.ER_BULLET_HARDWARE_OPENGL
else:
    physicsClient = p.connect(p.DIRECT)
    p_render = p.ER_TINY_RENDERER

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)  # 设置重力
planeId = p.loadURDF("plane.urdf")

# 加载箱子，并将其设置为固定基座
boxStartPos = [-5, 3, 0]
boxStartOrientation = p.getQuaternionFromEuler([np.pi/2, 0, 0])
# boxId = p.loadURDF("E:/HandleImage/box.urdf", boxStartPos, boxStartOrientation, useFixedBase=1, globalScaling=0.01)
boxId = p.loadURDF("E:/HandleImage/box.urdf", boxStartPos, boxStartOrientation, useFixedBase=1, globalScaling=0.01)
# 修改颜色为蓝色
p.changeVisualShape(boxId, -1, rgbaColor=[0, 0, 0.8, 1])
p.changeDynamics(boxId, -1, restitution=0.1, lateralFriction=1.0)

planePos, planeOrn = p.getBasePositionAndOrientation(planeId)
boxPos, boxOrn = p.getBasePositionAndOrientation(boxId)
boxPos = (boxPos[0], -boxPos[2], boxPos[1])

initial_rotation = R.from_quat(boxOrn)
rotation_y_then_z = R.from_euler('yz', [-np.pi/2, -np.pi/2], degrees=False)
combined_rotation = initial_rotation * rotation_y_then_z
final_quaternion = combined_rotation.as_quat()
boxOrn = final_quaternion

# boxOrn = p.getQuaternionFromEuler(boxOrn)
print(f'boxPos:{boxPos},boxOrn:{boxOrn}')
boxOrn = np.roll(boxOrn, shift=1)
print(f'boxOrn_wxyz:{boxOrn}')

handles = []  # 用于存储handle的ID
# 箱子内部随机放置n个handle
for i in range(1):
    # 随机生成位置和朝向
    startPos = [np.random.uniform(-4, 4), np.random.uniform(-2, 2), np.random.uniform(0.2, 0.5)]
    # startOrientation = p.getQuaternionFromEuler([np.random.uniform(-np.pi, np.pi) for _ in range(3)])
    # startPos = [0, 0, 2]
    startOrientation = p.getQuaternionFromEuler([np.pi/4, np.pi/2, 0])
    # initial_handle_rotation = R.from_quat(startOrientation)
    # rotation = R.from_euler('y', np.pi/2, degrees=False)
    # combined_handle_rotation = initial_handle_rotation * rotation
    # final_handle_quaternion = combined_handle_rotation.as_quat()
    # startOrientation = final_handle_quaternion

    # 载入handle
    handleId = p.loadURDF("E:/HandleImage/handle.urdf", startPos, startOrientation)
    metallic_color = [0.7, 0.7, 0.7, 1]  # RGBA，这里的透明度(A)设置为1（不透明）
    # p.changeVisualShape(handleId, -1, rgbaColor=metallic_color)
    # p.changeDynamics(handleId, -1, restitution=0.1, lateralFriction=1.0)
    p.changeDynamics(handleId, -1, mass=0)
    handles.append(handleId)

# 设置相机位置在箱子上方，例如在z轴上方2个单位处
cameraEyePosition = [0, 0, 10]

# 设置相机焦点在箱子中心
cameraTargetPosition = [0, 0, 4]

# 设置相机的向上向量，通常为Z轴正方向
cameraUpVector = [0, 1, 0]

# 设置相机视角
viewMatrix = p.computeViewMatrix(
    cameraEyePosition,
    cameraTargetPosition,
    cameraUpVector)

# 设置投影矩阵参数
width, height = 1280, 960
fov = 62.1  # 视野角度
aspect = width / height  # 宽高比
nearVal = 0.1  # 近裁剪面
farVal = 100  # 远裁剪面

# 根据FOV和图像尺寸计算焦距fx, fy
fx = width / (2.0 * np.tan(fov * np.pi / 360.0))
fy = fx # 假设fx=fy，对于方形像素
cx = width / 2
cy = height / 2
print("相机内参:")
print("fx =", fx, "fy =", fy)
print("cx =", cx, "cy =", cy)

projectionMatrix = p.computeProjectionMatrixFOV(
    fov=fov, aspect=aspect, nearVal=0.1, farVal=farVal)

lightDirection = [0, 0, 0]


def lookAt(eye, target, up):
    eye = np.array(eye, dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    up = np.array(up, dtype=np.float64)

    z_axis = eye - target
    if np.linalg.norm(z_axis) != 0:  # 避免除以0
        z_axis /= np.linalg.norm(z_axis)  # 归一化
    x_axis = np.cross(up, z_axis)
    if np.linalg.norm(x_axis) != 0:  # 避免除以0
        x_axis /= np.linalg.norm(x_axis)  # 归一化
    y_axis = np.cross(z_axis, x_axis)
    if np.linalg.norm(y_axis) != 0:  # 避免除以0
        y_axis /= np.linalg.norm(y_axis)  # 归一化

    mat = np.eye(4, dtype=np.float64)  # 使用浮点类型的单位矩阵
    mat[0][:3] = x_axis
    mat[1][:3] = y_axis
    mat[2][:3] = z_axis
    mat[:3, 3] = eye
    return np.linalg.inv(mat)  # 返回世界到相机坐标系的变换矩阵


def render(width, height, view_matrix, projection_matrix):
    """
    渲染图像
    """
    images = p.getCameraImage(width,
                              height,
                              view_matrix,
                              projection_matrix,
                              lightDirection=lightDirection,
                              shadow=True,
                              renderer=p_render)
    rgb = np.array(images[2]).reshape((height, width, 4))[:, :, :3]
    depth_raw = np.array(images[3]).reshape((height, width))
    seg = np.array(images[4]).reshape((height, width))

    far = farVal
    near = nearVal
    dep = far * near / (far - (far - near) * depth_raw)
    dep = np.asanyarray(dep).astype(np.float32) * 1000.
    dep = (dep.astype(np.uint16))
    dep = Image.fromarray(dep)

    return rgb, dep, seg


def _create_raymond_lights():
    # 定义三个光源的theta角度（与Z轴的夹角），以弧度表示
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])

    # 定义三个光源的phi角度（在XY平面的旋转角度），以弧度表示
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    # 初始化一个空列表，用于存放光源节点
    nodes = []

    # 遍历每对phi和theta角度
    for phi, theta in zip(phis, thetas):
        # 根据球坐标系转换公式计算光源方向向量的X分量
        xp = np.sin(theta) * np.cos(phi)
        # 根据球坐标系转换公式计算光源方向向量的Y分量
        yp = np.sin(theta) * np.sin(phi)
        # 根据球坐标系转换公式计算光源方向向量的Z分量
        zp = np.cos(theta)

        # 将计算得到的方向向量组合成数组
        z = np.array([xp, yp, zp])
        # 对方向向量进行归一化处理
        z = z / np.linalg.norm(z)
        # 计算与方向向量垂直的一个向量x
        x = np.array([-z[1], z[0], 0.0])
        # 如果x是零向量，则将其设置为默认值[1.0, 0.0, 0.0]
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        # 对x向量进行归一化处理
        x = x / np.linalg.norm(x)
        # 计算z和x的叉乘，得到第三个垂直的向量y
        y = np.cross(z, x)

        # 初始化一个4x4的单位矩阵
        matrix = np.eye(4)
        # 将x, y, z向量作为旋转矩阵的前三列，构建变换矩阵
        matrix[:3, :3] = np.c_[x, y, z]
        # 创建一个光源节点，设置光源类型为定向光，并应用之前计算的变换矩阵
        nodes.append(
            pyrender.Node(
                light=pyrender.DirectionalLight(color=np.ones(3), intensity=2.0),
                matrix=matrix,
            )
        )

    # 返回创建的光源节点列表
    return nodes


# def wait_for_objects_to_stabilize():
#     """
#     等待直到所有物体都静止。
#     """
#     while True:
#         all_objects_stable = True
#         num_bodies = p.getNumBodies()
#         for i in range(num_bodies):
#             # 获取物体的线速度和角速度
#             lin_vel, ang_vel = p.getBaseVelocity(i)
#             # 检查速度是否足够小
#             if np.linalg.norm(lin_vel) > 1e-4 or np.linalg.norm(ang_vel) > 1e-4:
#                 all_objects_stable = False
#                 break
#         if all_objects_stable:
#             break
#         else:
#             p.stepSimulation()


for i in range(480):
    p.stepSimulation()
    time.sleep(1./240.)
    # p.getCameraImage(width,
    #                  height,
    #                  viewMatrix,
    #                  projectionMatrix,
    #                  lightDirection=lightDirection,
    #                  shadow=True,
    #                  renderer=p.ER_BULLET_HARDWARE_OPENGL)

handlePos, handleOrn = p.getBasePositionAndOrientation(handleId)
handlePos = (handlePos[0], -handlePos[2], handlePos[1])
# handleOrn = p.getEulerFromQuaternion(handleOrn)
# handleOrn = (handleOrn[0], -handleOrn[2], handleOrn[1])
# handleOrn = p.getQuaternionFromEuler(handleOrn)
# initial_handle_rotation = R.from_quat(handleOrn)
# print(initial_handle_rotation)
# rotation_y_then_z = R.from_euler('yz', [-np.pi/2, -np.pi/2], degrees=False)
# combined_handle_rotation = initial_handle_rotation * rotation_y_then_z
# final_quaternion = combined_handle_rotation.as_quat()
# handleOrn = final_quaternion
handleOrn = np.roll(handleOrn, shift=1)
# handleOrn = p.getEulerFromQuaternion(handleOrn)
print(f'handlePos:{handlePos},handleOrn:{handleOrn}')


# 使用pyrendr进行渲染
if use_pyrender:
    scale_factor = 0.01
    handle_mesh = trimesh.load('E:/HandleImage/handle.obj')
    box_mesh = trimesh.load('E:/HandleImage/box.obj')
    handle_material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=1.0,  # 高金属度
        roughnessFactor=0.3,  # 较低的粗糙度以增加光泽
        baseColorFactor=[1, 1, 1, 1.0]  # 接近白色的基础颜色，RGBA格式
    )
    box_material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.5,
        roughnessFactor=0.5,
        baseColorFactor=[0, 0, 0.8, 1.0]
    )
    handle_mesh.apply_scale(scale_factor)
    box_mesh.apply_scale(scale_factor)
    handle_render_mesh = pyrender.Mesh.from_trimesh(handle_mesh, material=handle_material)  # 创建PyRender中的mesh对象
    box_render_mesh = pyrender.Mesh.from_trimesh(box_mesh,material=box_material)

    scene = pyrender.Scene()  # 创建PyRender场景
    box_node = pyrender.Node(mesh=box_render_mesh)  # 创建物体的节点并将其添加到场景中
    scene.add_node(box_node)
    handle_node = pyrender.Node(mesh=handle_render_mesh)  # 创建物体的节点并将其添加到场景中
    scene.add_node(handle_node)


    # 更新节点的位置
    box_transform_matrix = np.eye(4)
    box_transform_matrix[:3, :3] = np.array(p.getMatrixFromQuaternion(boxOrn)).reshape((3, 3))
    box_transform_matrix[:3, 3] = boxPos
    scene.set_pose(box_node, pose=box_transform_matrix)
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = np.array(p.getMatrixFromQuaternion(handleOrn)).reshape((3, 3))
    transform_matrix[:3, 3] = handlePos
    scene.set_pose(handle_node, pose=transform_matrix)

    # 创建透视相机
    # camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1280.0/960.0)
    camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy, nearVal, farVal)
    cn = pyrender.Node()
    cn.camera = camera
    # cam_pose = lookAt(cameraEyePosition, cameraTargetPosition, cameraUpVector)
    cam_pose = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, -12.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
    cam_pose[:, 1:3] *= -1.0
    cn.matrix = cam_pose
    print(cam_pose)
    scene.add_node(cn)

    # 创建定向光源，color是光源颜色，intensity是光强
    directional_light = pyrender.DirectionalLight(color=np.ones(3), intensity=5)
    # 光源的位置和朝向
    light_pose = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 1],
        [0, -1, 0, 0],
        [0, 0, 0, 1]
    ])
    light_pose[:, 1:3] *= -1
    # 将光源添加到场景中
    scene.add(directional_light, pose=light_pose)
    ray_light_nodes = _create_raymond_lights()
    [scene.add_node(rln) for rln in ray_light_nodes]

    pyrender.Viewer(scene, use_raymond_lighting=True)
    handle_pose = scene.get_pose(handle_node)
    print(handle_pose)
    # 创建OffscreenRenderer进行渲染
    renderer = pyrender.OffscreenRenderer(1280, 960)
    color, depth = renderer.render(scene)
    color_bgr = cv.cvtColor(color, cv.COLOR_RGB2BGR)
    cv.imwrite("E:/HandleImage/syndata/rgb.png", color_bgr)

# 渲染图像
# wait_for_objects_to_stabilize()
rgbImg, depthImg, segImg = render(width, height, viewMatrix, projectionMatrix)


# # segImg处理
# object_ids = np.unique(segImg)  # 获取所有唯一的物体ID
# object_ids = object_ids[2:]  # 假设0是背景，1是box
# masks = segImg == object_ids[:, None, None]  # 为每个物体ID创建掩模
# print(masks.shape)
#
# segmentation = []
# for i, mask in enumerate(masks):
#     # 将布尔掩模转换为uint8二值图像
#     mask_uint8 = (mask.astype(np.uint8) * 255)
#     contours, _ = cv.findContours(mask_uint8, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#     epsilon = 1.5
#     # contours 是一个列表，其中每个元素是轮廓的点集
#     for contour in contours:
#         # 使用轮廓近似来减少坐标点的数量
#         contour = cv.approxPolyDP(contour, epsilon, True)
#         # 将轮廓点扁平化成一维数组
#         contour = contour.flatten().tolist()
#         if len(contour) > 6:  # 确保轮廓至少包含3个点
#             segmentation.append(contour)
#
# for i, poly in enumerate(segmentation):
#     print(f"{i }:{[poly]}")


# 将RGB图像转换为BGR图像
rgbImg_bgr = cv.cvtColor(rgbImg, cv.COLOR_RGB2BGR)
cv.imwrite("E:/HandleImage/syndata/rgbImg.png", rgbImg_bgr)
# cv.imwrite("E:/HandleImage/syndata/depthImg.png", depthImg)
depthImg.save("E:/HandleImage/syndata/depthImg.png")
cv.imwrite("E:/HandleImage/syndata/segImg.png", segImg)
p.disconnect()

show = False
if show:
    # # 显示RGB图像
    plt.figure(figsize=(24, 8))
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.imshow(rgbImg)
    plt.title("RGB Image")

    plt.subplot(1, 3, 2)
    depthImg = np.array(depthImg)  # 将深度图像转换为numpy数组
    depthImg = depthImg / np.max(depthImg)  # 归一化深度值到0-1
    plt.imshow(depthImg, cmap='gray')  # 使用灰度色彩映射
    plt.title("Depth Image")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(segImg)
    plt.title("Segmentation Image")
    plt.axis('off')
    plt.show()

