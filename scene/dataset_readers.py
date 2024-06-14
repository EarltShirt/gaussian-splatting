#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    # positions = np.vstack([vertices['x']/100, vertices['y']/100, vertices['z']/100]).T
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    try:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    except:
        colors = np.zeros_like(positions)
    try:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except:
        normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos


#################################################################################################
####################################### MY ADDITIONS ############################################

def load_bounds(file_path):
    # load the bounds dictionary from the json file
    print(f'file_path : {file_path}')
    with open(file_path, 'r') as file:
        bounds = json.load(file)
    return bounds

def save_bounds(file_path, bounds):
    with open(file_path, 'w') as file:
        json.dump(bounds, file)

def segment_from_ply(bounds, num_pts, path, gaussians):
    ply_path = os.path.join(path, "points3d.ply") 
    part1 = []
    part2 = []
    part3 = []
    part4 = []
    for key in bounds.keys():
        if 'part1' in key:
            part1.append(key)
        elif 'part2' in key:
            part2.append(key)
        elif 'part3' in key:
            part3.append(key)
        elif 'part4' in key:
            part4.append(key)
    parts = []
    parts.append(part1)
    parts.append(part2)
    parts.append(part3)
    parts.append(part4)

    # print(f'Parts : {parts}\n')

    for bound in bounds.values():
        bound["min"] = np.array(bound["min"])
        bound["max"] = np.array(bound["max"])

    # epsilon = np.ones(3) * 10e-4

    # for bound in bounds.values():
    #     bound["min"] = np.array(bound["min"])/100 - epsilon
    #     bound["max"] = np.array(bound["max"])/100 + epsilon

    # for bound in bounds.values():
    #     tmp = bound["min"].copy()[0]
    #     bound["min"][0] = bound["min"][2]
    #     bound["min"][2] = tmp
    #     tmp = bound["max"].copy()[0]
    #     bound["max"][0] = bound["max"][2]
    #     bound["max"][2] = tmp

    # shiftx = 0.03
    # shifty = 0.60
    # shiftz = -0.005
    # scale = 0.60

    # for bound in bounds.values():
    #     bound["min"] = np.array([bound["min"][0] + shiftx, bound["min"][1] + shifty, bound["min"][2] + shiftz]) * scale
    #     bound["max"] = np.array([bound["max"][0] + shiftx, bound["max"][1] + shifty, bound["max"][2] + shiftz]) * scale

    
    # save it to tmp_bounds as a dictionnary of arrays 
    # tmp_bounds = {}
    # for key in bounds.keys():
    #     min = [bounds[key]["min"][0], bounds[key]["min"][1], bounds[key]["min"][2]]
    #     max = [bounds[key]["max"][0], bounds[key]["max"][1], bounds[key]["max"][2]]
    #     tmp_bounds[key] = {"min": min, "max": max}

    # save_bounds(bounds_path, tmp_bounds)    

    pcd = fetchPly(ply_path)
    points = pcd.points
    colors = pcd.colors
    normals = pcd.normals

    group_id = 0
    group_lst = np.zeros(num_pts)

    colors_lst = np.array([[255, 0, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255]])
    colors_lst = colors_lst / 255.0
    for part in parts:
        group_id += 1
        for subpart in part:
            min, max = np.array(bounds[subpart]["min"]), np.array(bounds[subpart]["max"])
            # print(f'\nGroup {group_id} - Part : {subpart} - Min : {min} - Max : {max}')
            i = 0
            j = 0
            for point in points:
                if np.all(point >= min) and np.all(point <= max):
                    group_lst[i] = group_id
                    colors[i] = colors_lst[group_id-1]
                    j += 1
                i += 1
            # print(f'Number of points in subpart {subpart} : {j}')
    # print(f'\nThe maximum height is {np.max(points[:,1])}\n')


    new_ply_path = os.path.join(path, "points3d_segmented.ply")
    new_pcd = BasicPointCloud(points=points, colors=colors, normals=normals)

    storePly(new_ply_path, new_pcd.points, SH2RGB(new_pcd.colors)*255)
    storePly(ply_path, new_pcd.points, SH2RGB(new_pcd.colors)*255)
    print(f'\nStoring segmented point cloud at {new_ply_path}')

    gaussians.set_bounds(bounds)   
    gaussians.set_groups(group_lst)
    gaussians.set_parts(parts)
    
    print(f'The minimum group id is {np.min(group_lst)} and the maximum group id is {np.max(group_lst)}\n')
    return new_pcd
    
def generate_pcd(bounds, num_pts, path, gaussians):
    part1 = []
    part2 = []
    part3 = []
    part4 = []
    num_pts_per_part = int(num_pts / len(bounds.keys())) # this is an int since there are 10 keys for the moment
    for key in bounds.keys():
        if 'part1' in key:
            part1.append(key)
        elif 'part2' in key:
            part2.append(key)
        elif 'part3' in key:
            part3.append(key)
        elif 'part4' in key:
            part4.append(key)
    parts = []
    parts.append(part1)
    parts.append(part2)
    parts.append(part3)
    parts.append(part4)
    xyz = np.empty(shape=(num_pts,3))
    # xyz = []
    i = 0
    group_id = 0
    group_lst = np.zeros(num_pts)
    for part in parts:
        group_id += 1
        for subpart in part:
            min, max = np.array(bounds[subpart]["min"]), np.array(bounds[subpart]["max"])
            xyz_tmp = np.random.uniform(low = min, high = max, size = (num_pts_per_part,3))
            xyz[num_pts_per_part*i:num_pts_per_part*(i+1), :] = xyz_tmp
            group_lst[num_pts_per_part*i:num_pts_per_part*(i+1)] = group_id
            # xyz.append(xyz_tmp)
            i+=1
    # xyz = np.array(xyz) # .reshape((num_pts,3))
    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    print(f'xyz shape : {np.shape(xyz)}')

    ply_path = os.path.join(path, "points3d.ply")
    storePly(ply_path, xyz, SH2RGB(shs) * 255)

    print(f'group list : {group_lst[::1000]}')

    gaussians.set_groups(group_lst)
    return pcd
#################################################################################################
#################################################################################################


def readNerfSyntheticInfo(path, white_background, eval, gaussians, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 300_000
        print(f"Generating random point cloud ({num_pts})...")

#################################################################################################
####################################### MY ADDITIONS ############################################
        bounds_path = os.path.join(path, "bounding_boxes.json")
        bounds = load_bounds(bounds_path)

        pcd = generate_pcd(bounds, num_pts, path, gaussians)
    
    
    if os.path.exists(ply_path):
        print(f'Point cloud found at {ply_path}')
        bounds_path = os.path.join(path, "bounding_boxes.json")
        bounds = load_bounds(bounds_path)
        num_points = len(fetchPly(ply_path).points)
        print(f'num_points : {num_points}')
        pcd = segment_from_ply(bounds, num_points, path, gaussians)
#################################################################################################
#################################################################################################

        # # Bounding_cube_size = 2.6
        # Bounding_cube_size = 20.0
        
        # # We create random points inside the bounds of the synthetic Blender scenes
        # xyz = np.random.random((num_pts, 3)) * Bounding_cube_size - (Bounding_cube_size / 2.0)
        # shs = np.random.random((num_pts, 3)) / 255.0
        # pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        # storePly(ply_path, xyz, SH2RGB(shs) * 255)

    # try:
    #     pcd = fetchPly(ply_path)
    # except:
    #     pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}