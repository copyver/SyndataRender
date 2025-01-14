"""设置物理引擎"""

import abc
import os
import time

import numpy as np
import pybullet
import trimesh
from autolab_core import Logger, RigidTransform
from pyrender import Mesh, Node, PerspectiveCamera, Scene, Viewer
from .utils import GRAVITY, USE_GUI


class PhysicsEngine(metaclass=abc.ABCMeta):
    """物理引擎抽象类"""

    def __init__(self):

        # set up logger
        self._logger = Logger.get_logger(self.__class__.__name__)

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def step(self):
        pass

    @abc.abstractmethod
    def start(self):
        pass

    @abc.abstractmethod
    def stop(self):
        pass


class PybulletPhysicsEngine(PhysicsEngine):
    """pybullet物理引擎的包装器，绑定到单个ID"""

    def __init__(self, urdf_cache_dir, debug=False):
        PhysicsEngine.__init__(self)
        self._physics_client = None
        self._debug = debug
        self._urdf_cache_dir = urdf_cache_dir
        if not os.path.isabs(self._urdf_cache_dir):
            self._urdf_cache_dir = os.path.join(
                os.getcwd(), self._urdf_cache_dir
            )

    def add(self, obj, static=False, globalScaling=1):

        # 构造URDF文件名，根据对象的键（key）来命名
        KEY_SEP_TOKEN = "~"
        urdf_filename = os.path.join(
            self._urdf_cache_dir,
            KEY_SEP_TOKEN.join(obj.key.split(KEY_SEP_TOKEN)[:-1]),
            "{}.urdf".format(
                KEY_SEP_TOKEN.join(obj.key.split(KEY_SEP_TOKEN)[:-1])
            ),
        )
        urdf_filename = os.path.normpath(urdf_filename)
        urdf_dir = os.path.dirname(urdf_filename)
        if not os.path.exists(urdf_filename):
            try:
                os.makedirs(urdf_dir)
            except:
                self._logger.warning(
                    "Failed to create dir %s. The object may have been created simultaneously by another process"
                    % (urdf_dir)
                )
            self._logger.info(
                "Exporting URDF for object {}".format(
                    KEY_SEP_TOKEN.join(obj.key.split(KEY_SEP_TOKEN)[:-1])
                )
            )

            # 修正对象的质心（为了渲染）和密度，然后导出为URDF文件
            geometry = obj.mesh.copy()
            geometry.apply_translation(-obj.mesh.center_mass)
            trimesh.exchange.export.export_urdf(geometry, urdf_dir)

        com = obj.mesh.center_mass
        pose = self._convert_pose(obj.pose, com)  # 转换对象的位姿，考虑质心的位置
        obj_t = pose.translation
        obj_q_wxyz = pose.quaternion  # 对象的四元数表示（wxyz格式）
        obj_q_xyzw = np.roll(obj_q_wxyz, -1)  # 转换四元数到xyzw格式，以满足PyBullet的要求

        # 使用PyBullet加载URDF文件，添加对象到仿真环境
        try:
            obj_id = pybullet.loadURDF(
                urdf_filename,
                obj_t,
                obj_q_xyzw,
                useFixedBase=static,
                globalScaling=globalScaling,
                physicsClientId=self._physics_client
            )
            # if USE_GUI:
            if obj.key == 'bin~0':
                pybullet.changeVisualShape(obj_id, -1, rgbaColor=[0, 0, 0.7, 1])
                num_joints = pybullet.getNumJoints(obj_id)
                for joint_index in range(num_joints):
                    pybullet.changeVisualShape(obj_id, joint_index, rgbaColor=[0, 0, 0.7, 1])
            if obj.key.split('~')[0] == 'handle':
                pybullet.changeVisualShape(obj_id, -1, rgbaColor=[0.67, 0.67, 0.71, 0])
                num_joints = pybullet.getNumJoints(obj_id)
                for joint_index in range(num_joints):
                    pybullet.changeVisualShape(obj_id, joint_index, rgbaColor=[0.67, 0.71, 0.71, 0])
            if obj.key.split('~')[0] == 'socket':
                pybullet.changeVisualShape(obj_id, -1, rgbaColor=[0.67, 0.67, 0.71, 0])
                num_joints = pybullet.getNumJoints(obj_id)
                for joint_index in range(num_joints):
                    pybullet.changeVisualShape(obj_id, joint_index, rgbaColor=[0.67, 0.71, 0.71, 0])
        except:
            raise Exception("Failed to load %s" % (urdf_filename))

        if self._debug:
            self._add_to_scene(obj)

        # 更新两个映射：对象键到PyBullet中对象ID的映射，和对象键到质心的映射
        self._key_to_id[obj.key] = obj_id
        self._key_to_com[obj.key] = com

    def get_velocity(self, key):
        obj_id = self._key_to_id[key]
        return pybullet.getBaseVelocity(
            obj_id, physicsClientId=self._physics_client
        )

    def get_pose(self, key):
        obj_id = self._key_to_id[key]
        obj_t, obj_q_xyzw = pybullet.getBasePositionAndOrientation(
            obj_id, physicsClientId=self._physics_client
        )
        obj_q_wxyz = np.roll(obj_q_xyzw, 1)
        pose = RigidTransform(
            rotation=obj_q_wxyz,
            translation=obj_t,
            from_frame="obj",
            to_frame="world",
        )
        pose = self._deconvert_pose(pose, self._key_to_com[key])
        return pose

    def remove(self, key):
        obj_id = self._key_to_id[key]
        pybullet.removeBody(obj_id, physicsClientId=self._physics_client)
        self._key_to_id.pop(key)
        self._key_to_com.pop(key)
        if self._debug:
            self._remove_from_scene(key)

    def step(self):
        pybullet.stepSimulation(physicsClientId=self._physics_client)
        if self._debug:
            time.sleep(0.04)
            self._update_scene()

    def reset(self):
        if self._physics_client is not None:
            self.stop()
        self.start()

    def start(self):
        if self._physics_client is None:
            if not USE_GUI:
                self._physics_client = pybullet.connect(pybullet.DIRECT)
            else:
                self._physics_client = pybullet.connect(pybullet.GUI)
                pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
            pybullet.setGravity(
                0, 0, GRAVITY, physicsClientId=self._physics_client
            )
            self._key_to_id = {}
            self._key_to_com = {}
            if self._debug:
                self._create_scene()
                self._viewer = Viewer(
                    self._scene, use_raymond_lighting=True, run_in_thread=True
                )

    def stop(self):
        if self._physics_client is not None:
            pybullet.disconnect(self._physics_client)
            self._physics_client = None
            if self._debug:
                self._scene = None
                self._viewer.close_external()
                while self._viewer.is_active:
                    pass

    def __del__(self):
        self.stop()
        del self

    def _convert_pose(self, pose, com):
        new_pose = pose.copy()
        new_pose.translation = pose.rotation.dot(com) + pose.translation
        return new_pose

    def _deconvert_pose(self, pose, com):
        new_pose = pose.copy()
        new_pose.translation = pose.rotation.dot(-com) + pose.translation
        return new_pose

    def _create_scene(self):
        self._scene = Scene()
        camera = PerspectiveCamera(
            yfov=0.833, znear=0.05, zfar=3.0, aspectRatio=1.0
        )
        cn = Node()
        cn.camera = camera
        pose_m = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.88],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        pose_m[:, 1:3] *= -1.0
        cn.matrix = pose_m
        self._scene.add_node(cn)
        self._scene.main_camera_node = cn

    def _add_to_scene(self, obj):
        self._viewer.render_lock.acquire()
        n = Node(
            mesh=Mesh.from_trimesh(obj.mesh),
            matrix=obj.pose.matrix,
            name=obj.key,
        )
        self._scene.add_node(n)
        self._viewer.render_lock.release()

    def _remove_from_scene(self, key):
        self._viewer.render_lock.acquire()
        if self._scene.get_nodes(name=key):
            self._scene.remove_node(
                next(iter(self._scene.get_nodes(name=key)))
            )
        self._viewer.render_lock.release()

    def _update_scene(self):
        self._viewer.render_lock.acquire()
        for key in self._key_to_id.keys():
            obj_pose = self.get_pose(key).matrix
            if self._scene.get_nodes(name=key):
                next(iter(self._scene.get_nodes(name=key))).matrix = obj_pose
        self._viewer.render_lock.release()
