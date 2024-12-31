from synenv import scenesyndata as ssdata

if __name__ == '__main__':
    # 生成合成数据
    scene_images = ssdata.load_images_from_folder('E:/HandleImage/poseDataset/scene_img/')
    object_images = ssdata.load_images_from_folder('E:/HandleImage/poseDataset/train/')
    object_masks = ssdata.load_images_from_folder('E:/HandleImage/poseDataset/train_mask')
    output_fodler = 'E:/HandleImage/poseDataset/syndata/'
    ssdata.generate_synthetic_data(scene_images, object_images, object_masks, 20, 10, output_fodler)
    # 生成掩模文件
    ann_filepath = 'E:/HandleImage/poseDataset/syndata/annotations.json'
    img_dirpath = 'E:/HandleImage/poseDataset/syndata/'
    mask_dirpath = 'E:/HandleImage/poseDataset/syndata_mask/'
    ssdata.generate_groupmask_images(ann_filepath, img_dirpath, mask_dirpath)
