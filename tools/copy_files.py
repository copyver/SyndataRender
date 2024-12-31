import shutil
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm


def copy_file(source_dest_tuple):
    """
    复制单个文件到目标目录。
    :param source_dest_tuple: (source_file, dest_dir)
    """
    source_file, dest_dir = source_dest_tuple
    try:
        dest_file = Path(dest_dir) / source_file.name
        shutil.copy2(source_file, dest_file)  # 保留文件元数据
        return source_file  # 返回成功的文件路径用于显示进度
    except Exception as e:
        print(f"Failed to copy {source_file}: {e}")
        return None


def copy_folder_multiprocess(source_dir, dest_dir, num_processes=4):
    """
    使用多进程复制文件夹中的所有文件到目标文件夹，并显示进度条。
    """
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)

    # 确保目标目录存在
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 获取源文件夹中所有文件
    files = [file for file in source_dir.iterdir() if file.is_file()]

    # 准备 (source_file, dest_dir) 参数对
    tasks = [(file, dest_dir) for file in files]

    # 使用多进程池进行文件复制
    with Pool(processes=num_processes) as pool:
        # 使用 tqdm 显示进度
        for _ in tqdm(pool.imap(copy_file, tasks), total=len(tasks), desc="Copying Files"):
            pass


if __name__ == "__main__":
    # 示例：将 source 文件夹下的文件复制到 dest 文件夹
    source_folder = ("/home/yhlever/DeepLearning/6D_object_pose_estimation/Datasets"
                     "/HANDLE/datasets_handle_4000t_800v/val/images/color_ims")  # 源文件夹路径
    dest_folder = "/home/yhlever/DeepLearning/6D_object_pose_estimation/Datasets/handle-seg/images/val"  # 目标文件夹路径

    copy_folder_multiprocess(source_folder, dest_folder, num_processes=24)
