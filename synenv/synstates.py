"""
定义了几个类，用于表示仿真环境中不同对象的状态,用于机器人视觉系统，
每个类代表环境中的一个元素，如物体、堆、相机等。
"""
import numpy as np
import logging


class State(object):
    """状态抽象类"""
    pass


class ObjectState(State):
    """
    单个物体
    Attributes
    ----------
    key : str
        标识物体的字符串
    mesh : URDF
        存储物体的几何形状
    pose : RigidTransform
        描述物体在世界坐标系中的位置和方向
    sim_id : int
        物体在仿真环境中的ID
    """
    def __init__(self, key, mesh, pose=None, sim_id=-1):
        self.key = key
        self.mesh = mesh
        self.pose = pose
        self.sim_id = sim_id


class HeapState(State):
    """
    物体对象以及基底对象（箱子和平面）堆叠状态
    Attributes
    ----------
    workspace_states obj_states: list of ObjectState
    """
    def __init__(self, workspace_states, obj_states, metadata={}):
        self.workspace_states = workspace_states
        self.obj_states = obj_states
        self.metadata = metadata

    @property
    def workspace_keys(self):
        return [s.key for s in self.workspace_states]

    @property
    def workspace_meshes(self):
        return [s.mesh for s in self.workspace_states]

    @property
    def workspace_sim_ids(self):
        return [s.sim_id for s in self.workspace_states]

    @property
    def obj_keys(self):
        return [s.key for s in self.obj_states]

    @property
    def obj_meshes(self):
        return [s.mesh for s in self.obj_states]

    @property
    def obj_sim_ids(self):
        return [s.sim_id for s in self.obj_states]

    @property
    def num_objs(self):
        return len(self.obj_keys)

    def __getitem__(self, key):
        return self.state(key)

    def state(self, key):
        try:
            return self.obj_states[self.obj_keys.index(key)]
        except:
            try:
                return self.workspace_states[self.workspace_keys.index(key)]
            except:
                logging.warning("Object %s not in pile!")
        return None


class CameraState(State):
    """相机状态.
    Attributes
    ----------
    mesh : Trimesh
        物体几何的三角形网格表示
    pose : RigidTransform
        相机相对于世界的姿态
    intrinsics : CameraIntrinsics
        透视投影模型中摄像机的特性
    """

    def __init__(self, frame, pose, intrinsics):
        self.frame = frame
        self.pose = pose
        self.intrinsics = intrinsics

    @property
    def height(self):
        return self.intrinsics.height

    @property
    def width(self):
        return self.intrinsics.width

    @property
    def aspect_ratio(self):
        return self.width / float(self.height)

    @property
    def yfov(self):
        return 2.0 * np.arctan(self.height / (2.0 * self.intrinsics.fy))


class HeapAndCameraState(object):
    """物体堆和相机的状态."""

    def __init__(self, heap_state, cam_state):
        self.heap = heap_state
        self.camera = cam_state

    @property
    def obj_keys(self):
        return self.heap.obj_keys

    @property
    def num_objs(self):
        return self.heap.num_objs


