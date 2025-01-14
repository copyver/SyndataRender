#!/bin/bash

export CAD_PATH="/home/yhlever/DeepLearning/SyndataRender/meshes/objects/socket/socket05.ply"
export OUTPUT_DIR="templates/socket05"
export CNOS_CAM_FPATH="predefined_poses/cam_poses_level0.npy"
export NORMALIZE="True"
export COLORIZE="False"

# 2) 通过 BlenderProc 运行 Python 脚本
blenderproc run render_template.py
