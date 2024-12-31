import os
import shutil


def check_and_clear_directory(dir_path, logger):
    # 检查目录是否存在，如果不存在则创建
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logger.info("目录不存在，已创建新目录。继续运行程序。")
        return

    # 检查目录中是否存在文件
    if os.listdir(dir_path):
        logger.info("存在旧数据集，是否覆盖？(y/n):")
        choice = input().strip().lower()  # 获取用户输入，并转换为小写
        if choice == 'y':
            # 清空文件夹下所有文件
            for filename in os.listdir(dir_path):
                file_path = os.path.join(dir_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    logger.info(f'删除 {file_path} 时发生了错误。原因: {e}')
            logger.info("旧数据集已清空。")
        elif choice == 'n':
            logger.info("保留旧数据集，继续运行程序。")
        else:
            logger.info("输入无效，请输入'y'或'n'。")
    else:
        logger.info("目录为空，继续运行程序。")


def obj_to_camera_pose(R_c, t_c, R_o, t_o):
    """输入相机相对于世界的位姿和对象相对于世界的位姿，输出对象相对于相机的位姿"""
    # 计算从世界坐标系到相机坐标系的旋转矩阵和平移向量
    R_wc = R_c.T  # 世界到相机的旋转矩阵是相机旋转矩阵的转置

    # 将对象的位姿从世界坐标系变换到相机坐标系
    R_oc = R_wc @ R_o  # 对象相对于相机的旋转矩阵
    t_oc = R_wc @ (t_o - t_c)  # 对象相对于相机的平移向量

    return R_oc, t_oc


# constants
USE_GUI = 0
GRAVITY = -9.8
