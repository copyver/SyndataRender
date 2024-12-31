import trimesh

stl = "E:/HandleImage/handle.stl"
mesh = trimesh.load_mesh(stl)
urdf_dir = "D:/DeepLearning/hf_maskrcnn/meshes/handle"
mesh.apply_translation(-mesh.center_mass)
trimesh.exchange.export.export_urdf(mesh, urdf_dir)
