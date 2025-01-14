import blenderproc as bproc

import os
import json
import cv2
import numpy as np
import trimesh


def get_norm_info(mesh_path):
    mesh = trimesh.load(mesh_path, force='mesh')

    model_points = trimesh.sample.sample_surface(mesh, 1024)[0]
    model_points = model_points.astype(np.float32)

    min_value = np.min(model_points, axis=0)
    max_value = np.max(model_points, axis=0)

    radius = max(np.linalg.norm(max_value), np.linalg.norm(min_value))

    return 1 / (2 * radius)

if __name__ == '__main__':
    # set relative path of Data folder
    tless_path = '/home/yhlever/DeepLearning/6D_object_pose_estimation/datasets/tless'
    output_dir = os.path.join(tless_path, 'templates')
    cnos_cam_fpath = "predefined_poses/cam_poses_level0.npy"

    bproc.init()

    model_path = os.path.join(tless_path, 'models_cad')
    models_info = json.load(open(os.path.join(model_path, 'models_info.json')))
    for obj_id in models_info.keys():
        diameter = models_info[obj_id]['diameter']
        scale = 1 / diameter
        obj_fpath = os.path.join(model_path, f'obj_{int(obj_id):06d}.ply')
        # scale = get_norm_info(obj_fpath)

        cam_poses = np.load(cnos_cam_fpath)
        for idx, cam_pose in enumerate(cam_poses[:]):

            bproc.clean_up()

            # load object
            obj = bproc.loader.load_obj(obj_fpath)[0]
            obj.set_scale([scale, scale, scale])
            obj.set_cp("category_id", obj_id)

            color = [0.4, 0.4, 0.4, 0.]
            material = bproc.material.create('obj')
            material.set_principled_shader_value('Base Color', color)
            obj.set_material(0, material)

            # convert cnos camera poses to blender camera poses
            cam_pose[:3, 1:3] = -cam_pose[:3, 1:3]
            cam_pose[:3, -1] = cam_pose[:3, -1] * 0.001 * 2
            bproc.camera.add_camera_pose(cam_pose)

            # set light
            light_energy = 1000
            light_scale = 2.5
            light1 = bproc.types.Light()
            light1.set_type("POINT")
            light1.set_location(
                [light_scale * cam_pose[:3, -1][0], light_scale * cam_pose[:3, -1][1], light_scale * cam_pose[:3, -1][2]])
            light1.set_energy(light_energy)

            bproc.renderer.set_max_amount_of_samples(1)
            # render the whole pipeline
            data = bproc.renderer.render()
            # render nocs
            data.update(bproc.renderer.render_nocs())

            # check save folder
            save_fpath = os.path.join(output_dir, f'obj_{int(obj_id):06d}')
            if not os.path.exists(save_fpath):
                os.makedirs(save_fpath)

            # save rgb image
            color_bgr_0 = data["colors"][0]
            color_bgr_0[..., :3] = color_bgr_0[..., :3][..., ::-1]
            cv2.imwrite(os.path.join(save_fpath, 'rgb_' + str(idx) + '.png'), color_bgr_0)

            # save mask
            mask_0 = data["nocs"][0][..., -1]
            cv2.imwrite(os.path.join(save_fpath, 'mask_' + str(idx) + '.png'), mask_0 * 255)

            # save nocs
            xyz_0 = 2 * (data["nocs"][0][..., :3] - 0.5)
            np.save(os.path.join(save_fpath, 'xyz_' + str(idx) + '.npy'), xyz_0.astype(np.float16))
