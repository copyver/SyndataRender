"""定义虚拟环境"""

import json

import cv2
import gym
import numpy as np
from pyrender import (
    DirectionalLight,
    IntrinsicsCamera,
    Mesh,
    MetallicRoughnessMaterial,
    Node,
    OffscreenRenderer,
    RenderFlags,
    Scene,
    Viewer,
)

from .pybullet_engine import PybulletPhysicsEngine
from .synspaces import HeapAndCameraStateSpace
import synenv.scenesyndata as ssdata
from.utils import obj_to_camera_pose
from autolab_core import Logger
from datetime import datetime


class BinHeapEnv(gym.Env):
    """OpenAI Gym-style环境创建箱中的对象堆"""

    def __init__(self, config):

        self._config = config

        # 读取配置
        self._state_space_config = self._config["state_spaces"]

        # 初始化类变量
        self._state = None
        self._scene = None
        self._physics_engine = PybulletPhysicsEngine(
            urdf_cache_dir=config["urdf_cache_dir"], debug=config["debug"]
        )
        self._state_space = HeapAndCameraStateSpace(
            self._physics_engine, self._state_space_config
        )

        self._categories = config["categories"]
        self._camera_states = {}  # 用于存储所有相机状态的字典
        self._reset_count = 0  # 用于记录重置次数的计数器
        self._images_info = []  # 用于存储所有图像数据的列表
        self._annotations_info = []  # 存储标注信息的列表
        self._image_id = None
        self._annotation_id = 0
        self._obj_pose_data = {}  # 存储每个掩模相对于相机的位姿

    @property
    def categories(self):
        return self._categories

    @property
    def camera_states(self):
        return self._camera_states

    @property
    def reset_count(self):
        return self._reset_count

    @property
    def images_info(self):
        return self._images_info

    @property
    def annotations_info(self):
        return self._annotations_info

    @property
    def image_id(self):
        return self._image_id

    @property
    def annotation_id(self):
        return self._annotation_id

    @property
    def obj_pose_data(self):
        return self._obj_pose_data

    @property
    def config(self):
        return self._config

    @property
    def state(self):
        return self._state

    @property
    def camera(self):
        return self._camera

    @property
    def observation(self):
        return self.render_camera_image()

    @property
    def scene(self):
        return self._scene

    @property
    def num_objects(self):
        return self.state.num_objs

    @property
    def state_space(self):
        return self._state_space

    @property
    def obj_keys(self):
        return self.state.obj_keys

    @image_id.setter
    def image_id(self, value):
        self._image_id = value

    @reset_count.setter
    def reset_count(self, value):
        self._reset_count = value

    @annotation_id.setter
    def annotation_id(self, value):
        self._annotation_id = value

    def _up_annotation_id(self):
        self.annotation_id += 1

    def _reset_state_space(self):
        """采样一个新的静态和动态状态."""
        state = self._state_space.sample()
        self._state = state.heap
        self._camera = state.camera

    def _update_scene(self):
        # 更新相机
        camera = IntrinsicsCamera(
            self.camera.intrinsics.fx,
            self.camera.intrinsics.fy,
            self.camera.intrinsics.cx,
            self.camera.intrinsics.cy,
        )
        cn = next(iter(self._scene.get_nodes(name=self.camera.frame)))
        cn.camera = camera
        pose_m = self.camera.pose.matrix.copy()
        pose_m[:, 1:3] *= -1.0
        cn.matrix = pose_m
        self._scene.main_camera_node = cn

        # 更新工作区
        for obj_key in self.state.workspace_keys:
            next(
                iter(self._scene.get_nodes(name=obj_key))
            ).matrix = self.state[obj_key].pose.matrix

        # 更新对象
        for obj_key in self.state.obj_keys:
            next(
                iter(self._scene.get_nodes(name=obj_key))
            ).matrix = self.state[obj_key].pose.matrix

    def _reset_scene(self, scale_factor=1.0):
        """重置场景.

        Parameters
        ----------
        scale_factor : float
            optional scale factor to apply to the image dimensions
        """
        # delete scene
        if self._scene is not None:
            self._scene.clear()
            del self._scene

        # create scene
        scene = Scene()

        # setup camera
        camera = IntrinsicsCamera(
            fx=self.camera.intrinsics.fx,
            fy=self.camera.intrinsics.fy,
            cx=self.camera.intrinsics.cx,
            cy=self.camera.intrinsics.cy,
            znear=0.05,  # 近裁面
            zfar=100.0  # 远裁面
        )
        pose_m = self.camera.pose.matrix.copy()
        pose_m[:, 1:3] *= -1.0
        scene.add(camera, pose=pose_m, name=self.camera.frame)
        scene.main_camera_node = next(
            iter(scene.get_nodes(name=self.camera.frame))
        )


        workspace_material = MetallicRoughnessMaterial(
            metallicFactor=0,
            roughnessFactor=1,
            baseColorFactor=[10, 10, 10, 255]
        )

        object_material = MetallicRoughnessMaterial(
            metallicFactor=0.8,  # 高金属度
            roughnessFactor=0.4,  # 较低的粗糙度以增加光泽
            baseColorFactor=[157, 167, 163, 255]  # 接近白色的基础颜色，RGBA格式
        )

        # 添加工作区对象
        for obj_key in self.state.workspace_keys:
            if obj_key == 'bin~0':
                material = workspace_material
            else:
                material = None
            obj_state = self.state[obj_key]
            obj_mesh = Mesh.from_trimesh(obj_state.mesh, material=material)
            T_obj_world = obj_state.pose.matrix
            scene.add(obj_mesh, pose=T_obj_world, name=obj_key)

        # 添加场景对象
        for obj_key in self.state.obj_keys:
            obj_state = self.state[obj_key]
            obj_mesh = Mesh.from_trimesh(obj_state.mesh, material=object_material)
            T_obj_world = obj_state.pose.matrix
            scene.add(obj_mesh, pose=T_obj_world, name=obj_key)

        # 添加光(用于显色)
        light = DirectionalLight(color=np.ones(3), intensity=1.0)
        scene.add(light, pose=np.eye(4))
        ray_light_nodes = self._create_raymond_lights()
        [scene.add_node(rln) for rln in ray_light_nodes]

        self._scene = scene

    def reset_camera(self):
        """只重置相机.
        Useful for generating image data for multiple camera views
        """
        self._camera = self.state_space.camera.sample()
        self._update_scene()

        k_matrix = self._camera.intrinsics.proj_matrix.flatten().tolist()  # 将内参矩阵转换为列表
        self.camera_states[str(self.reset_count)] = {
            "cam_K": k_matrix,
            "depth_scale": 1
        }

        # 更新重置次数
        self.reset_count += 1

    def save_camera_states_to_json(self, filepath):
        """将所有相机状态写入JSON文件。
        Parameters
        ----------
        filepath : str
            输出文件的路径。
        """
        with open(filepath, 'w') as f:
            json.dump(self.camera_states, f, indent=2)

    def save_obj_pose_to_json(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.obj_pose_data, f, indent=2)

    def save_obj_instances_to_json(self, filepath):

        info = {
            "description": "lhy 2025 Dataset",
            "url": "",
            "version": "1.0",
            "year": 2025,
            "contributor": "lin haiyang",
            "date_created": datetime.now().isoformat()
        }

        categories_info = []
        for name, id in self.categories.items():
            category = {
                "supercategory": "industrial",
                "id": id,
                "name": name
            }
            categories_info.append(category)

        coco_format = {
            "info": info,
            "images": self.images_info,
            "annotations": self.annotations_info,
            "categories": categories_info
        }
        with open(filepath, 'w') as f:
            json.dump(coco_format, f, indent=2)

    def reset(self):
        """重置环境."""

        # reset state space
        self._reset_state_space()

        # reset scene
        self._reset_scene()

    def view_3d_scene(self):
        """在3D查看器中渲染场景."""
        if self.state is None or self.camera is None:
            raise ValueError(
                "Cannot render 3D scene before state is set! You can set the state with the reset() function"
            )

        Viewer(self.scene, use_raymond_lighting=True)

    def render_camera_image(self, color=True):
        """渲染当前场景的相机图像."""
        renderer = OffscreenRenderer(self.camera.width, self.camera.height)
        # flags = RenderFlags.NONE if color else RenderFlags.DEPTH_ONLY
        flags = RenderFlags.SHADOWS_DIRECTIONAL if color else RenderFlags.DEPTH_ONLY
        image = renderer.render(self._scene, flags=flags)
        renderer.delete()
        if color:
            file_name = "image_{:06d}.png".format(self.image_id)
            image_info = ssdata.create_image_info(self.image_id, file_name, self.camera.width, self.camera.height)
            self.images_info.append(image_info)
        return image

    def render_segmentation_images(self):
        """
        为处于场景中的每个对象渲染分割掩模(模态与非模态).

        1. 获取整张图的深度信息 full_depth.
        2. 为每个对象分别渲染:
           - amodal_mask(全遮挡掩模).
           - modal_mask(可见掩模).
        3. 判断是否被遮挡(iscrowd)，并为未被遮挡物体生成相应标注信息和姿态信息.
        4. 返回 amodal_data, modal_data，分别包含所有对象的非模态与模态掩模叠加信息.
        """
        # 1) 渲染出整张图的深度信息，以便后续进行可见性(模态)判断
        full_depth = self.render_camera_image(color=False)  # shape: (H, W)

        # 提前获取 H, W，便于后续使用
        height, width = full_depth.shape[:2]

        # 构建用于保存所有对象 amodal 和 modal 掩模数据的数组
        # 形状: (H, W, num_objects)，每一层对应一个对象
        num_objects = len(self.obj_keys)
        amodal_data = np.zeros((height, width, num_objects), dtype=np.uint8)
        modal_data = np.zeros((height, width, num_objects), dtype=np.uint8)

        # 创建离屏渲染器，设置仅渲染深度
        renderer = OffscreenRenderer(viewport_width=width, viewport_height=height)
        flags = RenderFlags.DEPTH_ONLY

        # 获取当前场景中所有对象对应的 mesh node
        obj_mesh_nodes = []
        for k in self.obj_keys:
            node_list = list(self._scene.get_nodes(name=k))
            if len(node_list) == 0:
                raise ValueError(f"No node found for object key: {k}")
            obj_mesh_nodes.append(node_list[0])

        # 2) 隐藏场景中所有 mesh
        for mesh_node in self._scene.mesh_nodes:
            mesh_node.mesh.is_visible = False

        # 遍历每个对象节点，单独渲染它的深度，得到 amodal_mask 和 modal_mask
        for i, node in enumerate(obj_mesh_nodes):
            # 只显示当前对象
            node.mesh.is_visible = True

            # 渲染当前对象深度
            depth = renderer.render(self._scene, flags=flags)

            # 全覆盖掩模: 只要有深度，说明对象存在 (被遮挡也算)
            amodal_mask = depth > 0.0

            # 模态掩模: 需要与 full_depth 相对应且深度几乎一致的点
            # 若对象出现在整张 full_depth 的可见表面上，则深度相等(或几乎相等)
            modal_mask = np.logical_and(
                np.abs(depth - full_depth) < 1e-6,
                full_depth > 0.0
            )

            # 将掩模填充到对应的通道中
            amodal_data[amodal_mask, i] = 255
            modal_data[modal_mask, i] = 255

            # 恢复为不可见，继续处理下一个对象
            node.mesh.is_visible = False

        # 渲染结束后删除离屏渲染器
        renderer.delete()

        # 3) 显示场景中所有 mesh
        for mesh_node in self._scene.mesh_nodes:
            mesh_node.mesh.is_visible = True

        # 4) 遍历对象掩模，判断遮挡关系，并收集标注及姿态信息
        self.obj_pose_data[str(self.image_id)] = []

        for i, obj_key in enumerate(self.obj_keys):
            # amodal 与 modal 掩模
            mask_amodal = amodal_data[:, :, i]  # 全遮挡掩模
            mask_modal = modal_data[:, :, i]  # 可见掩模

            # 遮挡判断：amodal mask 中的像素数大于 modal mask，说明有被遮挡部分
            if np.sum(mask_amodal) - np.sum(mask_modal) > 50:
                iscrowd = 1
            else:
                iscrowd = 0

            # 若被遮挡(iscrowd=1)可根据业务需求选择处理或跳过
            if iscrowd == 0:
                # 仅处理未被遮挡物体(也可根据业务场景改成同时处理)
                # 这一步可自行调整逻辑：此处示例直接拿 modal_mask 来做最终的可见掩模
                final_mask = (mask_modal > 0).astype(np.uint8) * 255

                # 生成分割与 bounding box 信息
                segmentation = ssdata.generate_segmentation(final_mask)
                bbox = ssdata.get_bbox_from_mask(final_mask)

                # 根据 obj_key 提取 category_id
                category_key = obj_key.split('~')[1]
                category_id = self._categories.get(category_key)
                if category_id is None:
                    raise ValueError(f"{obj_key} not found in categories")

                # 创建并保存标注信息
                annotation = ssdata.create_annotation(
                    segmentation,
                    bbox,
                    iscrowd,
                    self.image_id,
                    category_id=category_id,
                    annotation_id=self.annotation_id
                )
                self.annotations_info.append(annotation)

                # 获取姿态信息(物体相对世界坐标 pose)
                m2w_pose = self._scene.get_pose(obj_mesh_nodes[i])
                r_m2w, t_m2w = m2w_pose[:3, :3], m2w_pose[:3, 3]

                # 获取相机姿态信息(相机相对世界坐标 pose)
                r_c2w = self._camera.pose.rotation
                t_c2w = self._camera.pose.translation

                # 计算物体相对于相机的姿态
                r_m2c, t_m2c = obj_to_camera_pose(r_c2w, t_c2w, r_m2w, t_m2w)
                r_m2c = r_m2c.flatten().tolist()
                t_m2c = t_m2c.flatten().tolist()

                # 记录每个对象的姿态信息
                pose_info = {
                    "cam_R_m2c": r_m2c,
                    "cam_t_m2c": t_m2c,
                    "annotation_id": self.annotation_id
                }
                self.obj_pose_data[str(self.image_id)].append(pose_info)

                # 更新全局注释计数
                self._up_annotation_id()

        # 5) 返回每个对象的叠加 amodal_data 和 modal_data
        return amodal_data, modal_data

    def _create_raymond_lights(self):
        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

        nodes = []

        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            yp = np.sin(theta) * np.sin(phi)
            zp = np.cos(theta)

            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)

            matrix = np.eye(4)
            matrix[:3, :3] = np.c_[x, y, z]
            nodes.append(
                Node(
                    light=DirectionalLight(color=np.ones(3), intensity=1.0),
                    matrix=matrix,
                )
            )

        return nodes
