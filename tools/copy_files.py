import shutil
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm


def copy_file(args):
    """
    复制单个文件到目标目录，保留文件元数据。
    :param args: (source_file, relative_path, dest_dir)
    """
    source_file, relative_path, dest_dir = args
    try:
        # 在目标文件夹中创建对应的子目录结构
        dest_file = Path(dest_dir) / relative_path
        dest_file.parent.mkdir(parents=True, exist_ok=True)

        # 复制文件并保留元数据
        shutil.copy2(source_file, dest_file)
        return source_file
    except Exception as e:
        print(f"Failed to copy {source_file}: {e}")
        return None


def copy_folder_multiprocess(source_dir, dest_dir, num_processes=4):
    """
    使用多进程复制文件夹中的所有文件和子文件夹到目标文件夹。
    :param source_dir: 源文件夹路径
    :param dest_dir: 目标文件夹路径
    :param num_processes: 进程数
    """
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 递归查找源文件夹中的所有文件（包括子文件夹）
    # 如果需要包含文件夹，也可以加上 file.is_dir() 的操作，这里只需要复制文件即可
    all_files = [file for file in source_dir.rglob('*') if file.is_file()]

    # 准备 (source_file, relative_path, dest_dir) 参数对
    tasks = []
    for file in all_files:
        relative_path = file.relative_to(source_dir)
        tasks.append((file, relative_path, dest_dir))

    with Pool(processes=num_processes) as pool:
        for _ in tqdm(pool.imap_unordered(copy_file, tasks), total=len(tasks), desc="Copying Files"):
            pass


if __name__ == "__main__":
    source_folder = "/path/to/source"
    dest_folder = "/path/to/destination"

    copy_folder_multiprocess(source_folder, dest_folder, num_processes=8)
