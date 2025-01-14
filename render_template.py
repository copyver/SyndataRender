
import blenderproc as bproc  # must be first import
import os
import time
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


def main():
    tic = time.time()

    bproc.init()

    # load cnos camera pose
    cam_poses = np.load(cnos_cam_fpath)

    # calculating the scale of CAD model
    if normalize:
        scale = get_norm_info(cad_path)
    else:
        scale = 1

    for idx, cam_pose in enumerate(cam_poses):

        bproc.clean_up()

        # load object
        obj = bproc.loader.load_obj(cad_path)[0]
        obj.set_scale([scale, scale, scale])
        obj.set_cp("category_id", 1)

        # assigning material colors to untextured objects
        if colorize:
            material = bproc.material.create('obj')
            material.set_principled_shader_value('Base Color', color)
            material.set_principled_shader_value('Metallic', 0.8)
            material.set_principled_shader_value('Roughness', 0.6)
            obj.set_material(0, material)

        # convert cnos camera poses to blender camera poses
        cam_pose[:3, 1:3] = -cam_pose[:3, 1:3]
        cam_pose[:3, -1] = cam_pose[:3, -1] * 0.001 * 2
        bproc.camera.add_camera_pose(cam_pose)

        # set light
        light_scale = 2.5
        light_energy = 50000
        light1 = bproc.types.Light()
        light1.set_type("POINT")
        light1.set_location(
            [light_scale * cam_pose[:3, -1][0], light_scale * cam_pose[:3, -1][1], light_scale * cam_pose[:3, -1][2]])
        light1.set_energy(light_energy)

        bproc.renderer.set_max_amount_of_samples(50)
        # render the whole pipeline
        data = bproc.renderer.render()
        # render nocs
        data.update(bproc.renderer.render_nocs())

        # check save folder
        save_fpath = output_dir
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

    print("Done in {:.2f}s".format(time.time() - tic))


if __name__ == '__main__':
    # cad_path = "/home/yhlever/DeepLearning/SyndataRender/meshes/objects/handle/handle.obj"
    # output_dir = "templates/handle"
    # normalize = True
    # colorize = False
    # cnos_cam_fpath = "predefined_poses/cam_poses_level0.npy"
    color = [0.67, 0.71, 0.71, 0.]
    cad_path = os.getenv("CAD_PATH", "/path/to/default/handle.obj")
    output_dir = os.getenv("OUTPUT_DIR", "templates/handle")
    cnos_cam_fpath = os.getenv("CNOS_CAM_FPATH", "predefined_poses/cam_poses_level0.npy")
    normalize = os.getenv("NORMALIZE", "True") == "True"
    colorize = os.getenv("COLORIZE", "False") == "True"
    main()
