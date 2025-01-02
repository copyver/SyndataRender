"""渲染数据集"""

import argparse
import gc
import json
import os
import time
import traceback

import autolab_core.utils as utils
import matplotlib.pyplot as plt
import numpy as np
import yaml
from autolab_core import (
    BinaryImage,
    ColorImage,
    DepthImage,
    GrayscaleImage,
    Logger,
)
from synenv.utils import check_and_clear_directory
from synenv.synenv import BinHeapEnv
from synenv.scenesyndata import mask_to_coco

logger = Logger.get_logger("./generate_syndata.py")

SEED = 7


def generate_syndata(config):
    """渲染图像场景，生成数据集"""
    # 是否调试
    debug = config["debug"]
    if debug:
        np.random.seed(SEED)

    # 读取基础配置
    output_filename = config['output_filename']
    num_state = config['num_states']
    num_images_per_state = config['num_images_per_state']
    vis_config = config["vis"]

    max_objs_per_state = config["state_spaces"]["heap"]["objects"]["num_objects_per_images"]["max"]

    # 读取图像配置
    image_config = config['images']
    segmask_channels = max_objs_per_state + 1

    # 读取相机配置
    camera_config = config['state_spaces']['camera']
    im_width = camera_config['im_width']
    im_height = camera_config['im_height']

    imgdir = os.path.join(output_filename, 'images')
    check_and_clear_directory(imgdir, logger)

    # 设置输出目录
    if not os.path.isabs(output_filename):
        output_filename = os.path.join(os.getcwd(), output_filename)
    if not os.path.exists(output_filename):
        os.mkdir(output_filename)

    image_dir = os.path.join(output_filename, "images")
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)
    color_dir = os.path.join(image_dir, "color_ims")
    if image_config["color"] and not os.path.exists(color_dir):
        os.mkdir(color_dir)
    depth_dir = os.path.join(image_dir, "depth_ims")
    if image_config["depth"] and not os.path.exists(depth_dir):
        os.mkdir(depth_dir)
    amodal_dir = os.path.join(image_dir, "amodal_masks")
    if image_config["amodal"] and not os.path.exists(amodal_dir):
        os.mkdir(amodal_dir)
    modal_dir = os.path.join(image_dir, "modal_masks")
    if image_config["modal"] and not os.path.exists(modal_dir):
        os.mkdir(modal_dir)
    semantic_dir = os.path.join(image_dir, "semantic_masks")
    if image_config["semantic"] and not os.path.exists(semantic_dir):
        os.mkdir(semantic_dir)

    # 设置日志目录
    experiment_log_filename = os.path.join(
        output_filename, "dataset_generation.log"
    )
    if os.path.exists(experiment_log_filename):
        os.remove(experiment_log_filename)
    Logger.add_log_file(logger, experiment_log_filename, global_log_file=True)

    env = BinHeapEnv(config)
    obj_id_map = env.state_space.obj_id_map
    obj_keys = env.state_space.obj_keys
    obj_splits = env.state_space.obj_splits
    mesh_filenames = env.state_space.mesh_filenames
    save_obj_id_map = obj_id_map.copy()
    save_obj_id_map["environment"] = np.iinfo(np.uint32).max
    reverse_obj_ids = utils.reverse_dictionary(save_obj_id_map)

    # 保存元数据
    metadata = {}
    metadata["obj_ids"] = reverse_obj_ids
    metadata["obj_splits"] = obj_splits
    metadata["meshes"] = mesh_filenames
    json.dump(
        metadata,
        open(os.path.join(output_filename, "metadata.json"), "w"),
        indent=2,
        sort_keys=True,
    )

    state_id = 0
    while state_id < num_state:
        # 创建环境
        create_start = time.time()
        env = BinHeapEnv(config)
        env.state_space.obj_id_map = obj_id_map
        env.state_space.obj_keys = obj_keys
        env.state_space.set_splits(obj_splits)
        env.state_space.mesh_filenames = mesh_filenames
        create_stop = time.time()
        logger.info(
            "创建环境用时 %.3f 秒" % (create_stop - create_start)
        )

        states_remaining = num_state - state_id
        for i in range(states_remaining):
            logger.info("State: %06d" % state_id)

            try:
                env.reset()
                state = env.state
                split = state.metadata["split"]

                # 渲染状态
                if vis_config["state"]:
                    env.view_3d_scene()

                # 渲染图像
                for k in range(num_images_per_state):
                    env.image_id = num_images_per_state * state_id + k
                    # 重置相机
                    if num_images_per_state > 1:
                        env.reset_camera()
                    # 渲染图像
                    obs = env.render_camera_image(color=image_config["color"])
                    if image_config["color"]:
                        color_obs, depth_obs = obs
                    else:
                        depth_obs = obs

                    # 需要可视化则显示
                    if vis_config["obs"]:
                        if image_config["depth"]:
                            plt.figure()
                            plt.imshow(depth_obs)
                            plt.title("Depth Observation")
                        if image_config["color"]:
                            plt.figure()
                            plt.imshow(color_obs)
                            plt.title("Color Observation")
                        plt.show()

                    if (
                            image_config["modal"]
                            or image_config["amodal"]
                            or image_config["semantic"]
                    ):

                        # 渲染掩模分割图像
                        (
                            amodal_segmasks,
                            modal_segmasks,
                        ) = env.render_segmentation_images()

                        # 初始化掩码数组，用最大值填充表示背景
                        modal_segmask_arr = np.iinfo(np.uint8).max * np.ones(
                            [im_height, im_width, segmask_channels],
                            dtype=np.uint8,
                        )
                        amodal_segmask_arr = np.iinfo(np.uint8).max * np.ones(
                            [im_height, im_width, segmask_channels],
                            dtype=np.uint8,
                        )
                        stacked_segmask_arr = np.zeros(
                            [im_height, im_width, 1], dtype=np.uint8
                        )

                        # 将渲染得到的局部模态和全模态分割掩码填充到对应的数组中
                        modal_segmask_arr[
                        :, :, : env.num_objects
                        ] = modal_segmasks
                        amodal_segmask_arr[
                        :, :, : env.num_objects
                        ] = amodal_segmasks

                        # 如果配置中指定需要语义分割
                        if image_config["semantic"]:
                            # 对每个对象，将局部模态分割掩码中对象的像素设置为对象的索引（加1以避免与背景混淆）
                            for j in range(env.num_objects):
                                this_obj_px = np.where(
                                    modal_segmasks[:, :, j] > 0
                                )
                                stacked_segmask_arr[
                                    this_obj_px[0], this_obj_px[1], 0
                                ] = (j + 1)
                    # 可视化
                    if vis_config["semantic"]:
                        plt.figure()
                        plt.imshow(stacked_segmask_arr.squeeze())
                        plt.show()

                    # 保存图像和语义分割掩模
                    if image_config["color"]:
                        ColorImage(color_obs).save(
                            os.path.join(
                                color_dir,
                                "image_{:06d}.png".format(
                                    num_images_per_state * state_id + k
                                ),
                            )
                        )
                    if image_config["depth"]:
                        DepthImage(depth_obs).save(
                            os.path.join(
                                depth_dir,
                                "image_{:06d}.png".format(
                                    num_images_per_state * state_id + k
                                ),
                            )
                        )
                    if image_config["modal"]:
                        modal_id_dir = os.path.join(
                            modal_dir,
                            "image_{:06d}".format(
                                num_images_per_state * state_id + k
                            ),
                        )
                        if not os.path.exists(modal_id_dir):
                            os.mkdir(modal_id_dir)
                        for i in range(env.num_objects):
                            BinaryImage(modal_segmask_arr[:, :, i]).save(
                                os.path.join(
                                    modal_id_dir,
                                    "channel_{:03d}.png".format(i),
                                )
                            )
                    if image_config["amodal"]:
                        amodal_id_dir = os.path.join(
                            amodal_dir,
                            "image_{:06d}".format(
                                num_images_per_state * state_id + k
                            ),
                        )
                        if not os.path.exists(amodal_id_dir):
                            os.mkdir(amodal_id_dir)
                        for i in range(env.num_objects):
                            BinaryImage(amodal_segmask_arr[:, :, i]).save(
                                os.path.join(
                                    amodal_id_dir,
                                    "channel_{:03d}.png".format(i),
                                )
                            )
                    if image_config["semantic"]:
                        GrayscaleImage(stacked_segmask_arr.squeeze()).save(
                            os.path.join(
                                semantic_dir,
                                "image_{:06d}.png".format(
                                    num_images_per_state * state_id + k
                                ),
                            )
                        )

                # 删除动作对象
                for obj_state in state.obj_states:
                    del obj_state
                del state  # 删除当前状态，减少内存使用
                gc.collect()  # 调用垃圾收集器，尝试回收已删除对象的内存
                # 更新状态id
                env.save_camera_states_to_json('./syndatasets/scene_camera.json')
                env.save_obj_pose_to_json('./syndatasets/scene_pose_gt.json')
                env.save_obj_instances_to_json('./syndatasets/scene_instances_gt.json')
                state_id += 1

            except Exception as e:
                # log an error
                logger.warning("Heap failed!")
                logger.warning("%s" % (str(e)))
                logger.warning(traceback.print_exc())
                if debug:
                    raise

                del env
                gc.collect()
                env = BinHeapEnv(config)
                env.state_space.obj_id_map = obj_id_map
                env.state_space.obj_keys = obj_keys
                env.state_space.set_splits(obj_splits)
                env.state_space.mesh_filenames = mesh_filenames

        del env
        gc.collect()

    logger.info(
        "生成 %d 份图像数据" % (state_id * num_images_per_state)
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate a training dataset for a HF Mask R-CNN"
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default=None,
        help="configuration file to use",
    )
    args = parser.parse_args()
    config_filename = args.cfg
    if not os.path.isabs(config_filename):
        config_filename = os.path.join(os.getcwd(), config_filename)

    with open(config_filename, 'r') as file:
        config = yaml.safe_load(file)
    start_time = time.time()
    generate_syndata(config)
    end_time = time.time()
    logger.info("生成渲染数据集用时 %.3f min" % ((end_time - start_time) / 60))
