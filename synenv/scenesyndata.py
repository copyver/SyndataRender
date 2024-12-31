import json
import os
import random
from collections import defaultdict
from datetime import datetime
from glob import glob

import cv2
import numpy as np


def load_images_from_folder(folder):
    """加载文件夹中的所有图片"""
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


def create_info():
    """创建标注文件信息"""
    return {
        "description": "my Dataset",
        "url": "",
        "version": "1.0",
        "year": 2024,
        "contributor": "yhlever",
        "date_created": datetime.utcnow().isoformat(' ')
    }


def create_licenses():
    """创建证书信息"""
    return {
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
        }


def create_annotation(segmentation, bbox, iscrowd, image_id, category_id, annotation_id):
    """创建单个物体的COCO格式标注"""
    return {
        "segmentation": segmentation,
        "area": bbox[2] * bbox[3],
        "iscrowd": iscrowd,
        "image_id": image_id,
        "bbox": bbox,
        "category_id": category_id,
        "id": annotation_id
    }


def create_image_info(image_id, file_name, width, height, license_id=1, coco_url="", flickr_url=""):
    """创建单个图像的COCO格式信息"""
    return {
        "id": image_id,
        "file_name": file_name,
        "width": width,
        "height": height,
        "date_captured": datetime.now().isoformat(),
        "license": license_id,
        "coco_url": coco_url,
        "flickr_url": flickr_url
    }


def create_categories_info():
    """创建COCO的'categories'格式信息"""
    return {
        "supercategory": "industrial",
        "id": 1,
        "name": "handle",
        "keypoints": [
            "head",
            "mid",
            "tail"
        ],
        "skeleton": [
            [
                1,
                2
            ],
            [
                2,
                3
            ]
        ]
    }


def extract_individual_masks(mask_image):
    """使用连通区域标记从掩模图像中提取独立的掩模分割"""
    individual_masks = []

    # 应用连通区域标记
    num_labels, labels_im = cv2.connectedComponents(mask_image)

    for label in range(1, num_labels):  # 忽略背景标签0
        # 为每个独立区域创建一个掩模
        individual_mask = np.uint8(labels_im == label) * 255
        individual_masks.append(individual_mask)

    return individual_masks


def combine_images(scene, object_cropped, mask_cropped, position):
    """将物体叠加到场景图片上"""

    # 将物体和掩模放置到场景图片上的随机位置
    scene_copy = scene.copy()
    scene_copy[position[1]:position[1]+mask_cropped.shape[0], position[0]:position[0]+mask_cropped.shape[1]] = \
        np.where(mask_cropped[:, :, None] == 255, object_cropped, scene[position[1]:position[1]+mask_cropped.shape[0], position[0]:position[0]+mask_cropped.shape[1]])

    return scene_copy, mask_cropped


def check_overlap(occupied_mask, object_mask, position):
    """检查新物体是否与已放置物体重叠"""
    y, x = position
    y_end = y + object_mask.shape[0]
    x_end = x + object_mask.shape[1]

    if y_end > occupied_mask.shape[0] or x_end > occupied_mask.shape[1]:
        return True  # 超出边界

    # 检查重叠
    overlap_area = occupied_mask[y:y_end, x:x_end]
    return np.any(overlap_area[object_mask == 255] == 255)


def generate_poisson_position(lambda_x, lambda_y, max_x, max_y):
    """
    基于泊松分布生成位置的函数
    Args:
        lambda_x (int): x坐标的泊松分布参数
        lambda_y (int): y坐标泊松分布参数
        max_x (int): x最大位置参数
        max_y (int): y最大位置参数
    Returns:
        [x,y] 位置
    """
    while True:
        x = np.random.poisson(lambda_x)
        y = np.random.poisson(lambda_y)

        if 0 <= x <= max_x and 0 <= y <= max_y:
            return [x, y]


def generate_segmentation(mask, position=(0, 0), epsilon=1.0):
    """从二值掩模生成分割信息"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    x_offset, y_offset = position
    for contour in contours:
        # 使用轮廓近似来减少坐标点的数量
        contour = cv2.approxPolyDP(contour, epsilon, True)
        # 将轮廓点的坐标调整到它们在整个场景中的位置
        contour = contour + [x_offset, y_offset]
        # 将轮廓点扁平化成一维数组
        contour = contour.flatten().tolist()
        if len(contour) > 6:  # 确保轮廓至少包含3个点
            segmentation.append(contour)
    return segmentation


def is_binary_image(image):
    """检查图像是否为二值图"""
    unique_values = np.unique(image)
    if len(unique_values) == 2 and set(unique_values).issubset({0, 255}):
        return True
    else:
        return False


def get_bbox_from_mask(mask):
    if not is_binary_image(mask):
        # 如果不是二值图，进行二值化
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    else:
        binary_mask = mask

    points = np.transpose(np.nonzero(binary_mask))

    # 计算边界框 [x_min, y_min, x_max, y_max]
    if points.size > 0:
        y_min, x_min = np.min(points, axis=0)
        y_max, x_max = np.max(points, axis=0)
        # 转换为[x, y, width, height]
        bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
    else:
        bbox = [0, 0, 0, 0]  # 如果掩模全黑，则边界框为0

    return bbox


def generate_synthetic_data(scene_images, object_images, object_masks, num_images, num_masks, output_folder):
    """生成合成数据和对应的COCO格式标注文件"""
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # coco标注信息
    annotations = []
    images = []
    annotation_id = 1

    for i in range(num_images):
        # 随机选择场景和物体
        scene = random.choice(scene_images)
        occupied_mask = np.zeros(scene.shape[:2], dtype=np.uint8)  # 记录已占用区域的掩模
        synthetic_image = scene.copy()
        synthetic_image_filename = f'synthetic_{i + 1}.png'

        for _ in range(num_masks):
            object_idx = random.randint(0, len(object_masks) - 1)
            object = object_images[object_idx]
            object_mask = object_masks[object_idx]

            if len(object_mask.shape) == 3 and object_mask.shape[2] == 3:
                object_mask = cv2.cvtColor(object_mask, cv2.COLOR_BGR2GRAY)

            # 提取独立的掩模分割并随机选择一个
            individual_masks = extract_individual_masks(object_mask)
            if not individual_masks:
                continue
            selected_mask = random.choice(individual_masks)

            # 提取掩模区域的边界
            ys, xs = np.where(selected_mask == 255)
            if len(xs) == 0 or len(ys) == 0:
                return None, None
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()

            # 提取掩模中的物体
            object_cropped = object[y_min:y_max + 1, x_min:x_max + 1]
            mask_cropped = selected_mask[y_min:y_max + 1, x_min:x_max + 1]

            max_attempts = 50  # 限制尝试次数以避免无限循环
            for _ in range(max_attempts):
                # 随机选择物体的位置
                x = random.randint(0, scene.shape[1] - object_cropped.shape[1])
                y = random.randint(0, scene.shape[0] - object_cropped.shape[0])
                position = (x, y)

                if not check_overlap(occupied_mask, mask_cropped, position):
                    # 生成物体的边界框
                    bbox = [x, y, mask_cropped.shape[1], mask_cropped.shape[0]]

                    # 生成物体的分割信息
                    segmentation = generate_segmentation(mask_cropped, position)

                    # 创建标注并添加到列表
                    annotation = create_annotation(segmentation, bbox, i + 1, 1, annotation_id)
                    annotations.append(annotation)
                    annotation_id += 1

                    # 位置合适，合成物体
                    synthetic_image, _ = combine_images(synthetic_image, object_cropped, mask_cropped, position)
                    occupied_mask[y:y + mask_cropped.shape[0], x:x + mask_cropped.shape[1]] = np.maximum(
                        occupied_mask[y:y + mask_cropped.shape[0], x:x + mask_cropped.shape[1]], mask_cropped)
                    break  # 成功放置物体，跳出循环

        # 合成图片并更新标注
        cv2.imwrite(os.path.join(output_folder, synthetic_image_filename), synthetic_image)
        # 添加图像信息
        image_info = create_image_info(i + 1, synthetic_image_filename, scene.shape[1], scene.shape[0])
        images.append(image_info)

    categories = [create_categories_info()]
    # 保存标注和图像信息到JSON文件
    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    with open(os.path.join(output_folder, 'annotations.json'), 'w') as file:
        json.dump(coco_format, file)


def generate_mask_images(ann_file, img_dir, mask_dir):
    # 读取标注文件
    with open(ann_file, 'r') as f:
        annotations = json.load(f)

    # 确保掩模目录存在
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    # 遍历每个标注生成掩模
    for ann in annotations['annotations']:
        # 获取图像文件名和分割信息
        img_id = ann['image_id']
        img_info = next((img for img in annotations['images'] if img['id'] == img_id), None)
        if img_info is None:
            continue

        img_filename = img_info['file_name']
        segments = ann['segmentation']

        # 读取对应的图像
        img_path = os.path.join(img_dir, img_filename)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # 创建一个空白掩模
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # 填充掩模
        for segment in segments:
            poly = np.array(segment, dtype=np.int32).reshape((-1, 2))
            cv2.fillPoly(mask, [poly], 255)

        # 保存掩模图像
        mask_filename = f'mask_{img_id}.png'
        cv2.imwrite(os.path.join(mask_dir, mask_filename), mask)


def generate_groupmask_images(ann_file, img_dir, mask_dir):
    # 读取标注文件
    with open(ann_file, 'r') as f:
        annotations = json.load(f)

    # 确保掩模目录存在
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    # 根据img_id分组标注
    grouped_annotations = defaultdict(list)
    for ann in annotations['annotations']:
        grouped_annotations[ann['image_id']].append(ann)

    # 遍历每组标注生成掩模
    for img_id, anns in grouped_annotations.items():
        # 获取图像信息
        img_info = next((img for img in annotations['images'] if img['id'] == img_id), None)
        if img_info is None:
            continue

        img_filename = img_info['file_name']
        img_path = os.path.join(img_dir, img_filename)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # 创建一个空白掩模
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # 填充掩模
        for ann in anns:
            for segment in ann['segmentation']:
                poly = np.array(segment, dtype=np.int32).reshape((-1, 2))
                cv2.fillPoly(mask, [poly], 255)

        # 保存掩模图像
        mask_filename = f'mask_{img_id}.png'
        cv2.imwrite(os.path.join(mask_dir, mask_filename), mask)


def mask_to_coco(img_dir, amodal_mask_dir, out_dir):
    """读取图像和对应掩模，生成coco标注数据"""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    outputfile = os.path.join(out_dir, 'annotations.json')

    # coco标注信息
    annotations = []
    images = []
    annotation_id = 1

    for img_path in glob(os.path.join(img_dir, '*.png')):
        image_id = os.path.splitext(os.path.basename(img_path))[0]
        image_id = int(image_id.split('_')[-1])
        file_name = os.path.basename(img_path)
        image_info = create_image_info(image_id, file_name, 1280, 960)
        images.append(image_info)

    for mask_dir in os.listdir(amodal_mask_dir):
        mask_paths = glob(os.path.join(amodal_mask_dir, mask_dir, '*.png'))
        image_id = int(mask_dir.split('_')[-1])
        for mask_path in mask_paths:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            segmention = generate_segmentation(mask)
            bbox = get_bbox_from_mask(mask)
            annotation = create_annotation(segmention, bbox, image_id, category_id=1, annotation_id=annotation_id)
            annotation_id += 1
            annotations.append(annotation)

    categories = [create_categories_info()]
    # 保存标注和图像信息到JSON文件
    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    with open(outputfile, 'w') as file:
        json.dump(coco_format, file)


if __name__ == '__main__':
    img_dir = "D:/DeepLearning/hf_maskrcnn/syndatasets/images/color_ims"
    amodal_mask_dir = "D:/DeepLearning/hf_maskrcnn/syndatasets/images/amodal_masks"
    output_dir = "D:/DeepLearning/hf_maskrcnn/syndatasets"
    mask_to_coco(img_dir, amodal_mask_dir, output_dir)



