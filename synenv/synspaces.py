"""定义对象空间"""

import os
import time

import gym
import numpy as np
import scipy.stats as sstats
import trimesh
from autolab_core import Logger, RigidTransform

from .camera_random import CameraRandomVariable
from .synstates import CameraState, HeapAndCameraState, HeapState, ObjectState


class CameraStateSpace(gym.Space):
    """使用gym创建相机的状态空间."""

    def __init__(self, config):
        self._config = config

        # read params
        self.frame = config["name"]

        # random variable for pose of camera
        self.camera_rv = CameraRandomVariable(config)

    def sample(self):
        """Sample a camera state."""
        pose, intrinsics = self.camera_rv.sample(size=1)
        return CameraState(self.frame, pose, intrinsics)


class HeapStateSpace(gym.Space):
    """对象堆的状态空间."""

    def __init__(self, physics_engine, config):

        self._physics_engine = physics_engine
        self._config = config

        # 设置记录仪
        self._logger = Logger.get_logger(self.__class__.__name__)

        # 读取配置
        obj_config = config["objects"]
        workspace_config = config["workspace"]

        self.num_objs_rv = sstats.poisson(obj_config["numobjects_perimages"]["mean"] - 1)
        self.max_objs = obj_config["numobjects_perimages"]["max"]
        self.min_objs = obj_config["numobjects_perimages"]["min"]
        self.replace = config["replace"]

        self.max_obj_diam = obj_config["max_diam"]
        self.drop_height = config["drop_height"]
        self.max_settle_steps = config["max_settle_steps"]
        self.mag_v_thresh = config["mag_v_thresh"]
        self.mag_w_thresh = config["mag_w_thresh"]

        # 定义状态空间可以到达的位置范围
        min_heap_center = np.array(config["center"]["min"])
        max_heap_center = np.array(config["center"]["max"])
        self.heap_center_space = gym.spaces.Box(
            min_heap_center, max_heap_center, dtype=np.float32
        )

        # 设置对象配置
        # 在平面中对象放置姿态的边界
        # 组织为[tx, ty, theta]，其中theta的单位是度
        min_obj_pose = np.r_[obj_config["planar_translation"]["min"], 0]
        max_obj_pose = np.r_[
            obj_config["planar_translation"]["max"], 2 * np.pi
        ]
        self.obj_planar_pose_space = gym.spaces.Box(
            min_obj_pose, max_obj_pose, dtype=np.float32
        )

        # 对象掉落方向的界限
        min_sph_coords = np.array([0.0, 0.0])
        max_sph_coords = np.array([2 * np.pi, np.pi])
        self.obj_orientation_space = gym.spaces.Box(
            min_sph_coords, max_sph_coords, dtype=np.float32
        )

        # 质心的边界
        delta_com_sigma = max(1e-6, obj_config["center_of_mass"]["sigma"])
        self.delta_com_rv = sstats.multivariate_normal(
            np.zeros(3), delta_com_sigma ** 2
        )

        self.obj_density = obj_config["density"]

        # 工作空间边界(用于检查边界外)
        min_workspace_trans = np.array(workspace_config["min"])
        max_workspace_trans = np.array(workspace_config["max"])
        self.workspace_space = gym.spaces.Box(
            min_workspace_trans, max_workspace_trans, dtype=np.float32
        )

        # 设置对象键和目录
        object_keys = []
        mesh_filenames = []
        self._train_pct = obj_config["train_pct"]
        num_objects = obj_config["num_objects"]
        self._mesh_dir = obj_config["mesh_dir"]
        # 如果_mesh_dir不是绝对路径，则转换为绝对路径
        if not os.path.isabs(self._mesh_dir):
            self._mesh_dir = os.path.join(os.getcwd(), self._mesh_dir)
        for root, _, files in os.walk(self._mesh_dir):
            dataset_name = os.path.basename(root)
            if dataset_name in obj_config["object_keys"].keys():
                for f in files:
                    filename, ext = os.path.splitext(f)
                    if ext.split(".")[
                        1
                    ] in trimesh.exchange.load.mesh_formats() and (
                        filename in obj_config["object_keys"][dataset_name]
                        or obj_config["object_keys"][dataset_name] == "all"
                    ):  # 检查文件扩展名是否为支持的网格格式，并且文件名符合配置要求
                        obj_key = "{}{}{}".format(
                            dataset_name, "~", filename
                        )  # 构造对象键，格式为"数据集名称+分隔符+文件名"
                        object_keys.append(obj_key)
                        mesh_filenames.append(os.path.join(root, f))

        inds = np.arange(len(object_keys))  # 生成一个与object_keys长度相同的索引数组
        np.random.shuffle(inds)
        self.all_object_keys = list(np.array(object_keys)[inds][:num_objects])  # 使用打乱后的索引选择object_keys，数量限制为num_objects
        all_mesh_filenames = list(np.array(mesh_filenames)[inds][:num_objects])  # 与all_object_keys相对应，获取对应的网格文件名列表
        # 根据_train_pct划分训练集和测试集
        self.train_keys = self.all_object_keys[
            : int(len(self.all_object_keys) * self._train_pct)
        ]
        self.test_keys = self.all_object_keys[
            int(len(self.all_object_keys) * self._train_pct):
        ]
        # 创建一个字典，映射每个对象键到一个唯一的ID（从1开始）
        self.obj_ids = dict(
            [(key, i + 1) for i, key in enumerate(self.all_object_keys)]
        )
        # 创建一个字典，映射每个对象键到对应的网格文件名
        self.mesh_filenames = {}
        [
            self.mesh_filenames.update({k: v})
            for k, v in zip(self.all_object_keys, all_mesh_filenames)
        ]
        # 检查训练集和测试集是否都至少有一个对象，如果不是则抛出异常
        if (len(self.test_keys) == 0 and self._train_pct < 1.0) or (
            len(self.train_keys) == 0 and self._train_pct > 0.0
        ):
            raise ValueError("Not enough objects for train/test split!")

    @property
    def obj_keys(self):
        return self.all_object_keys

    @obj_keys.setter
    def obj_keys(self, keys):
        self.all_object_keys = keys

    @property
    def num_objects(self):
        return len(self.all_object_keys)

    @property
    def obj_id_map(self):
        return self.obj_ids

    @obj_id_map.setter
    def obj_id_map(self, id_map):
        self.obj_ids = id_map

    @property
    def obj_splits(self):
        obj_splits = {}
        for key in self.all_object_keys:
            if key in self.train_keys:
                obj_splits[key] = 0
            else:
                obj_splits[key] = 1
        return obj_splits

    def set_splits(self, obj_splits):
        self.train_keys = []
        self.test_keys = []
        for k in obj_splits.keys():
            if obj_splits[k] == 0:
                self.train_keys.append(k)
            else:
                self.test_keys.append(k)

    def in_workspace(self, pose):
        """检查一个姿势是否在工作区中."""
        t_pose = pose.translation.astype(np.float32)
        return self.workspace_space.contains(t_pose)

    def sample(self):
        """从空间中采样一个状态
        Returns
        -------
        :obj:`HeapState`
            state of the object pile
        """

        # 启动物理引擎
        self._physics_engine.start()

        # 设置工作空间(箱子和平面）
        workspace_obj_states = []
        workspace_objs = self._config["workspace"]["objects"]
        for work_key, work_config in workspace_objs.items():

            # 使路径成为绝对路径
            mesh_filename = work_config["mesh_filename"]
            pose_filename = work_config["pose_filename"]

            if not os.path.isabs(mesh_filename):
                mesh_filename = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    mesh_filename,
                )
            if not os.path.isabs(pose_filename):
                pose_filename = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    pose_filename,
                )

            # 加载网格
            mesh = trimesh.load_mesh(mesh_filename)
            mesh.density = self.obj_density
            pose = RigidTransform.load(pose_filename)
            workspace_obj = ObjectState(
                "{}~0".format(work_key), mesh, pose
            )
            self._physics_engine.add(workspace_obj, static=True)
            workspace_obj_states.append(workspace_obj)

        # 采样
        train = True
        # 根据设定的训练集比例随机决定是从训练集还是测试集中采样
        if np.random.rand() > self._train_pct:
            train = False
            sample_keys = self.test_keys
            self._logger.info("从测试集中采样")
        else:
            sample_keys = self.train_keys
            self._logger.info("从训练集中采样")

        total_num_objs = len(sample_keys)

        # 根据预设的概率分布抽样得到本次需要抽取的对象数量
        num_objs = self.num_objs_rv.rvs(size=1)[0]
        num_objs = min(num_objs, self.max_objs)
        num_objs = max(num_objs, self.min_objs)
        obj_inds = np.random.choice(
            np.arange(total_num_objs), size=2 * num_objs, replace=self.replace
        )
        self._logger.info("共采样 %d 个对象" % (num_objs))

        # 从定义的堆中心空间中随机抽样得到堆中心位置
        heap_center = self.heap_center_space.sample()
        t_heap_world = np.array([heap_center[0], heap_center[1], 0])  # 堆中心位置，Z坐标设为0（因为是平面上的位置）
        self._logger.debug(
            "Sampled pile location: %.3f %.3f"
            % (t_heap_world[0], t_heap_world[1])
        )

        # 采样样本对象，质心，姿态
        objs_in_heap = []
        total_drops = 0
        # 当总投放次数小于目标对象数量的两倍且堆中对象数少于目标数量时，循环继续
        while total_drops < 2 * num_objs and len(objs_in_heap) < num_objs:
            obj_key = sample_keys[obj_inds[total_drops]]  # 根据打乱后的索引选取对象键
            obj_mesh = trimesh.load_mesh(self.mesh_filenames[obj_key])  # 加载对应的网格模型
            obj_mesh.visual = trimesh.visual.ColorVisuals(  # 设置网格模型的颜色为灰色
                obj_mesh, vertex_colors=(0.7, 0.7, 0.7, 1.0)
            )
            obj_mesh.density = self.obj_density
            # obj_state_key = "{}".format(obj_key)  # 为对象生成一个状态键
            obj_state_key = "{}~{}".format(obj_key, total_drops)
            obj = ObjectState(obj_state_key, obj_mesh)
            self._logger.info(obj_state_key)
            _, radius = trimesh.nsphere.minimum_nsphere(obj.mesh)  # 计算对象的最小包围球半径
            if 2 * radius > self.max_obj_diam:
                self._logger.warning("对象半径过大, 跳过 .....")
                total_drops += 1
                continue

            # 采样对象质心
            delta_com = self.delta_com_rv.rvs(size=1)
            center_of_mass = obj.mesh.center_mass + delta_com
            obj.mesh.center_mass = center_of_mass

            # 采样对象姿态
            obj_orientation = self.obj_orientation_space.sample()
            az = obj_orientation[0]
            elev = obj_orientation[1]
            T_obj_table = RigidTransform.sph_coords_to_pose(
                az, elev
            ).as_frames("obj", "world")

            # 采样对象平面位姿
            obj_planar_pose = self.obj_planar_pose_space.sample()
            theta = obj_planar_pose[2]
            R_table_world = RigidTransform.z_axis_rotation(theta)
            R_obj_drop_world = R_table_world.dot(T_obj_table.rotation)
            t_obj_drop_heap = np.array(
                [obj_planar_pose[0], obj_planar_pose[1], self.drop_height]
            )
            t_obj_drop_world = t_obj_drop_heap + t_heap_world
            obj.pose = RigidTransform(
                rotation=R_obj_drop_world,
                translation=t_obj_drop_world,
                from_frame="obj",
                to_frame="world",
            )

            self._physics_engine.add(obj)
            # 尝试获取对象的速度，如果失败，则移除对象并继续
            try:
                v, w = self._physics_engine.get_velocity(obj.key)
            except:
                self._logger.warning(
                    "无法获取对象 %s 的速度. 跳过 ..."
                    % (obj.key)
                )
                self._physics_engine.remove(obj.key)
                total_drops += 1
                continue

            objs_in_heap.append(obj)

            # 等待对象静止，设置一个时间限制
            wait = time.time()
            objects_in_motion = True
            num_steps = 0
            while objects_in_motion and num_steps < self.max_settle_steps:

                # 一步模拟
                self._physics_engine.step()

                # 检查速度
                max_mag_v = 0
                max_mag_w = 0
                objs_to_remove = set()
                for o in objs_in_heap:
                    try:
                        v, w = self._physics_engine.get_velocity(o.key)
                    except:
                        self._logger.warning(
                            "无法获取对象 %s 的速度. 跳过 ..."
                            % (o.key)
                        )
                        objs_to_remove.add(o)
                        continue
                    mag_v = np.linalg.norm(v)
                    mag_w = np.linalg.norm(w)
                    if mag_v > max_mag_v:
                        max_mag_v = mag_v
                    if mag_w > max_mag_w:
                        max_mag_w = mag_w

                # 删除无效对象
                for o in objs_to_remove:
                    self._physics_engine.remove(o.key)
                    objs_in_heap.remove(o)

                # 检查运动中的物体
                if (
                    max_mag_v < self.mag_v_thresh
                    and max_mag_w < self.mag_w_thresh
                ):
                    objects_in_motion = False

                num_steps += 1

            # 读取物体姿态
            objs_to_remove = set()
            for o in objs_in_heap:
                obj_pose = self._physics_engine.get_pose(o.key)
                o.pose = obj_pose.copy()

                # 如果对象在工作空间之外，则删除该对象
                if not self.in_workspace(obj_pose):
                    self._logger.warning(
                        "对象 {} 超出工作空间边界!".format(o.key)
                    )
                    objs_to_remove.add(o)

            # 删除无效对象
            for o in objs_to_remove:
                self._physics_engine.remove(o.key)
                objs_in_heap.remove(o)
                self._logger.info(
                    "对象 {} 已经从工作空间中移除!".format(o.key)
                )

            total_drops += 1
            self._logger.debug(
                "Waiting for zero velocity took %.3f sec"
                % (time.time() - wait)
            )

        # 停止物理引擎
        self._physics_engine.stop()

        # 添加堆状态的元数据并返回它
        metadata = {"split": 0}
        if not train:
            metadata["split"] = 1

        return HeapState(workspace_obj_states, objs_in_heap, metadata=metadata)


class HeapAndCameraStateSpace(gym.Space):
    """环境的状态空间."""

    def __init__(self, physics_engine, config):

        heap_config = config["heap"]
        cam_config = config["camera"]

        # 单个状态空间
        self.heap = HeapStateSpace(physics_engine, heap_config)
        self.camera = CameraStateSpace(cam_config)

    @property
    def obj_id_map(self):
        return self.heap.obj_id_map

    @obj_id_map.setter
    def obj_id_map(self, id_map):
        self.heap.obj_ids = id_map

    @property
    def obj_keys(self):
        return self.heap.obj_keys

    @obj_keys.setter
    def obj_keys(self, keys):
        self.heap.all_object_keys = keys

    @property
    def obj_splits(self):
        return self.heap.obj_splits

    def set_splits(self, splits):
        self.heap.set_splits(splits)

    @property
    def mesh_filenames(self):
        return self.heap.mesh_filenames

    @mesh_filenames.setter
    def mesh_filenames(self, fns):
        self.heap.mesh_filenames = fns

    def sample(self):
        """Sample a state."""
        # 采样单个状态
        heap_state = self.heap.sample()
        cam_state = self.camera.sample()

        return HeapAndCameraState(heap_state, cam_state)