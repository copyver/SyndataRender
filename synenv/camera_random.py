"""随机采样相机状态"""
import numpy as np
import scipy.stats as sstats
from autolab_core import CameraIntrinsics, RigidTransform, transformations
from autolab_core.utils import sph2cart


class CameraRandomVariable(object):
    """一个平面上的均匀分布的相机姿势和内参。摄像机的位置指向(0,0,0)"""
    def __init__(self, config):
        """初始化CameraRandomVariable.
        Parameters
        ----------
        config : autolab_core.YamlConfig
            配置包含随机变量的参数
        Notes
        -----
        需要配置的参数在“其他参数”中填写
        ----------
        focal_length :        相机焦距
            min : float
            max : float
        delta_optical_center: 光心从中性变化
            min : float
            max : float
        radius:               从相机到世界原点的距离
            min : float
            max : float
        azimuth:              相机的方位角(与x轴的角度)，以度为单位.
            min : float
            max : float
        elevation:            摄像机的仰角(与z轴的角度)，以度为单位.
            min : float
            max : float
        roll:                 相机的转动(关于视角方向的角度)，以度为单位.
            min : float
            max : float
        x:                    世界中心在x轴上的平移.
            min : float
            max : float
        y:                    世界中心在y轴上的平移.
            min : float
            max : float
        im_height : float     以像素为单位的图像高度.
        im_width : float      以像素为单位的图像宽度.
        """
        # read params
        self.config = config
        self._parse_config(config)

        self.frame = config["name"]

        # 设置随机变量
        # 相机
        self.focal_rv = sstats.uniform(
            loc=self.min_f, scale=self.max_f - self.min_f
        )
        self.cx_rv = sstats.uniform(
            loc=self.min_cx, scale=self.max_cx - self.min_cx
        )
        self.cy_rv = sstats.uniform(
            loc=self.min_cy, scale=self.max_cy - self.min_cy
        )

        # viewsphere
        self.rad_rv = sstats.uniform(
            loc=self.min_radius, scale=self.max_radius - self.min_radius
        )
        self.elev_rv = sstats.uniform(
            loc=self.min_elev, scale=self.max_elev - self.min_elev
        )
        self.az_rv = sstats.uniform(
            loc=self.min_az, scale=self.max_az - self.min_az
        )
        self.roll_rv = sstats.uniform(
            loc=self.min_roll, scale=self.max_roll - self.min_roll
        )

        # table translation
        self.tx_rv = sstats.uniform(
            loc=self.min_x, scale=self.max_x - self.min_x
        )
        self.ty_rv = sstats.uniform(
            loc=self.min_y, scale=self.max_y - self.min_y
        )

    def _parse_config(self, config):
        """将参数从配置读入类成员."""
        # camera params
        self.min_f = config["focal_length"]["min"]
        self.max_f = config["focal_length"]["max"]
        self.min_delta_c = config["delta_optical_center"]["min"]
        self.max_delta_c = config["delta_optical_center"]["max"]
        self.im_height = config["im_height"]
        self.im_width = config["im_width"]

        self.mean_cx = float(self.im_width - 1) / 2
        self.mean_cy = float(self.im_height - 1) / 2
        self.min_cx = self.mean_cx + self.min_delta_c
        self.max_cx = self.mean_cx + self.max_delta_c
        self.min_cy = self.mean_cy + self.min_delta_c
        self.max_cy = self.mean_cy + self.max_delta_c

        # viewsphere params
        self.min_radius = config["radius"]["min"]
        self.max_radius = config["radius"]["max"]
        self.min_az = np.deg2rad(config["azimuth"]["min"])
        self.max_az = np.deg2rad(config["azimuth"]["max"])
        self.min_elev = np.deg2rad(config["elevation"]["min"])
        self.max_elev = np.deg2rad(config["elevation"]["max"])
        self.min_roll = np.deg2rad(config["roll"]["min"])
        self.max_roll = np.deg2rad(config["roll"]["max"])

        # params of translation in plane
        self.min_x = config["x"]["min"]
        self.max_x = config["x"]["max"]
        self.min_y = config["y"]["min"]
        self.max_y = config["y"]["max"]

    def camera_to_world_pose(self, radius, elev, az, roll, x, y):
        """将球面坐标转换为相机在世界中的姿态."""
        # 从球面坐标生成相机中心
        delta_t = np.array([x, y, 0])
        camera_z = np.array([sph2cart(radius, az, elev)]).squeeze()
        camera_center = camera_z + delta_t
        camera_z = -camera_z / np.linalg.norm(camera_z)

        # 求经典摄像机的x和y轴
        camera_x = np.array([camera_z[1], -camera_z[0], 0])
        x_norm = np.linalg.norm(camera_x)
        if x_norm == 0:
            camera_x = np.array([1, 0, 0])
        else:
            camera_x = camera_x / x_norm
        camera_y = np.cross(camera_z, camera_x)
        camera_y = camera_y / np.linalg.norm(camera_y)

        # 如果需要，反转x方向，使y向下
        if camera_y[2] > 0:
            camera_x = -camera_x
            camera_y = np.cross(camera_z, camera_x)
            camera_y = camera_y / np.linalg.norm(camera_y)

        # rotate by the roll
        R = np.vstack((camera_x, camera_y, camera_z)).T
        roll_rot_mat = transformations.rotation_matrix(
            roll, camera_z, np.zeros(3)
        )[:3, :3]
        R = roll_rot_mat.dot(R)
        T_camera_world = RigidTransform(
            R, camera_center, from_frame=self.frame, to_frame="world"
        )

        return T_camera_world

    def sample(self, size=1):
        """从模型中抽样随机变量.
        Parameters
        ----------
        size : int
            取样数量
        Returns
        -------
        :obj:`list` of :obj:`CameraSample`
            采样相机的参数和姿势
        """
        samples = []
        for i in range(size):
            # 随机采样摄像机参数
            focal = self.focal_rv.rvs(size=1)[0]
            cx = self.cx_rv.rvs(size=1)[0]
            cy = self.cy_rv.rvs(size=1)[0]

            # 随机采样视场参数
            radius = self.rad_rv.rvs(size=1)[0]
            elev = self.elev_rv.rvs(size=1)[0]
            az = self.az_rv.rvs(size=1)[0]
            roll = self.roll_rv.rvs(size=1)[0]

            # 随机采样平面平移
            tx = self.tx_rv.rvs(size=1)[0]
            ty = self.ty_rv.rvs(size=1)[0]

            # 转换为姿态和内参
            pose = self.camera_to_world_pose(radius, elev, az, roll, tx, ty)
            intrinsics = CameraIntrinsics(
                self.frame,
                fx=focal,
                fy=focal,
                cx=cx,
                cy=cy,
                skew=0.0,
                height=self.im_height,
                width=self.im_width,
            )

            # 添加到采样
            samples.append((pose, intrinsics))

        # not a list if only 1 sample
        if size == 1:
            return samples[0]
        return samples
