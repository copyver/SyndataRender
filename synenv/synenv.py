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
        self._state_space_config = self._config["statespaces"]

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
            "depth_scale": 10
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

    def save_obj_instances_to_json(self,filepath):

        info = {
            "description": "lhy 2024 Dataset",
            "url": "",
            "version": "1.0",
            "year": 2024,
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
        flags = RenderFlags.SKIP_CULL_FACES if color else RenderFlags.DEPTH_ONLY
        image = renderer.render(self._scene, flags=flags)
        renderer.delete()
        if color:
            file_name = "image_{:06d}.png".format(self.image_id)
            image_info = ssdata.create_image_info(self.image_id, file_name, self.camera.width, self.camera.height)
            self.images_info.append(image_info)
        return image

    def render_segmentation_images(self):
        """为处于状态的每个对象渲染分割掩模(模态和非模态)."""
        full_depth = self.render_camera_image(color=False)
        modal_data = np.zeros(
            (full_depth.shape[0], full_depth.shape[1], len(self.obj_keys)),
            dtype=np.uint8,
        )
        amodal_data = np.zeros(
            (full_depth.shape[0], full_depth.shape[1], len(self.obj_keys)),
            dtype=np.uint8,
        )
        renderer = OffscreenRenderer(self.camera.width, self.camera.height)
        flags = RenderFlags.DEPTH_ONLY

        # 隐藏所有网格
        obj_mesh_nodes = [
            next(iter(self._scene.get_nodes(name=k))) for k in self.obj_keys
        ]
        for mn in self._scene.mesh_nodes:
            mn.mesh.is_visible = False

        for i, node in enumerate(obj_mesh_nodes):
            node.mesh.is_visible = True

            depth = renderer.render(self._scene, flags=flags)
            amodal_mask = depth > 0.0
            modal_mask = np.logical_and(
                (np.abs(depth - full_depth) < 1e-6), full_depth > 0.0
            )
            amodal_data[amodal_mask, i] = np.iinfo(np.uint8).max
            modal_data[modal_mask, i] = np.iinfo(np.uint8).max
            node.mesh.is_visible = False

        renderer.delete()

        # 显示所有网格
        for mn in self._scene.mesh_nodes:
            mn.mesh.is_visible = True

        self.obj_pose_data[str(self.image_id)] = []
        for i, obj_key in enumerate(self.obj_keys):
            # 获取单个物体的amodal和modal掩模
            amodal_mask = amodal_data[:, :, i]
            modal_mask = modal_data[:, :, i]

            # 检测遮挡：如果非模态掩码中的像素多于模态掩码，认为物体被遮挡
            if np.sum(amodal_mask) - np.sum(modal_mask) > 50:
                iscrowd = 1  # 标记为被遮挡
            else:
                iscrowd = 0  # 标记为未被遮挡

            # 仅处理未被遮挡的物体
            if iscrowd == 0:
                # 将掩模转换为uint8格式并处理
                amodal_mask_uint8 = (modal_mask > 0) * 255  # 使用modal掩模，因为这是未遮挡的掩模
                amodal_mask_uint8 = amodal_mask_uint8.astype(np.uint8)

                segmentation = ssdata.generate_segmentation(amodal_mask_uint8)
                bbox = ssdata.get_bbox_from_mask(amodal_mask_uint8)
                category_key = obj_key.split('~')[0]
                category_id = self._categories.get(category_key, None)
                if category_id is None:
                    raise ValueError(f"{obj_key} not found category id")

                annotation = ssdata.create_annotation(segmentation, bbox, iscrowd, self.image_id,
                                                      category_id=category_id,
                                                      annotation_id=self.annotation_id)
                self.annotations_info.append(annotation)

                m2w_pose = self._scene.get_pose(obj_mesh_nodes[i])  # 获取对应节点的姿态信息
                r_m2w = m2w_pose[:3, :3]
                t_m2w = m2w_pose[:3, 3]

                r_c2w = self._camera.pose.rotation  # 获取相机姿态信息
                t_c2w = self._camera.pose.translation

                r_m2c, t_m2c = obj_to_camera_pose(r_c2w, t_c2w, r_m2w, t_m2w)
                r_m2c = r_m2c.flatten().tolist()
                t_m2c = t_m2c.flatten().tolist()
                # 存储每个对象的轮廓和姿态信息
                pose = {
                    "cam_R_m2c": r_m2c,
                    "cam_t_m2c": t_m2c,
                    "annotation_id": self.annotation_id
                }
                self.obj_pose_data[str(self.image_id)].append(pose)
                self._up_annotation_id()

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
