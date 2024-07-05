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
import torch
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

#################################################################################################
####################################### MY ADDITIONS ############################################
import json
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision.io import read_image
from sklearn.decomposition import PCA


def load_bounds(file_path):
    # load the bounds dictionary from the json file
    with open(file_path, 'r') as file:
        bounds = json.load(file)
    return bounds


#################################################################################################
#################################################################################################


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, bounds_file, shs_checkpoint=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    only_fitting = False

    # if one only wants to fit the SH coefficients
    if shs_checkpoint is not None:
        (model_params, first_iter) = torch.load(shs_checkpoint)
        gaussians.restore(model_params, opt)
        only_fitting = True
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        return dataset, gaussians, scene, pipe, background
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving {} Gaussians".format(iteration, scene.gaussians.get_xyz.shape[0]))
                # print("Saving {} Gaussians".format(scene.gaussians.get_xyz.shape[0]))
                scene.save(iteration)

            if (iteration % 1000 == 0):
                print("\n[ITER {}] Number of Gaussians: {}".format(iteration, scene.gaussians.get_xyz.shape[0]))

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune_groups(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

#################################################################################################
####################################### MY ADDITIONS ############################################

            # Every 500 iterations, we regroup the gaussians using the bounding boxes
            # if iteration % 500 == 0 and iteration >= 2000 and iteration < 10000:
            #     print("\n[ITER {}] Regrouping Gaussians".format(iteration))
            #     scene.gaussians.regroup_and_prune()
            
            if iteration == 1000:
                print("\n[ITER {}] Group Visualization".format(iteration))
                # scene.gaussians.regroup_and_prune()
                segmented_ply_path = os.path.join(scene.model_path, "segmented.ply")    
                gaussians.save_segmented_ply(segmented_ply_path)

                # gaussians.regroup_and_prune()
                post_segmented_ply_path = os.path.join(scene.model_path, "post_segmented.ply")
                gaussians.save_post_segmented_ply(post_segmented_ply_path)

                print("\nStoring the pre-segmented point cloud at {}".format(segmented_ply_path))
                print("\nStoring the post-segmented point cloud at {}".format(post_segmented_ply_path))
                
            #     theta = -np.pi / 4
            #     ROT = [
            #         [1, 0, 0], 
            #         [0, np.cos(theta), np.sin(theta)], 
            #         [0, -np.sin(theta), np.cos(theta)] ]
            #     rotation_tensor = torch.tensor(ROT, dtype=torch.float, device="cpu")
            #     rotated_ply_path = os.path.join(scene.model_path, "rotated.ply")
            #     gaussians.store_rotated_groups(rotated_ply_path, 3, rotation_tensor)
            #     print("\nStoring the rotated point cloud at {}".format(rotated_ply_path))

            # if iteration == 25501:
            #     gaussians.regroup_and_prune()
            #     theta = - np.pi / 4
            #     # rotation_tensor = torch.tensor(ROT, dtype=torch.float, device="cuda")
            #     gaussians.rotate_gaussians(3, theta, axis='x')

            # if iteration == 25502:
            #     print("\n[ITER {}] Pausing the training for the user to check the results".format(iteration))
            #     input("Press Enter to continue...")
    print("\n[FINISHED] Saving Final Checkpoint")
    torch.save((gaussians.capture(), iteration), scene.model_path + "/final_chkpnt.pth") 
    gaussians.regroup_and_prune()
    return dataset, gaussians, scene, pipe, background

def shs_fit(args, dataset, gaussians, scene, pipe, background):
    if args.sh_fitting is None:
        chkpt_path = scene.model_path + "/final_chkpnt.pth"
    else:
        chkpt_path = args.sh_fitting

    folders_path = os.path.abspath(os.path.join(dataset.source_path, '..'))
    # folders_path = os.path(dataset.source_path).parent

    print(f'Loading the angles from the dataset at {folders_path}')
    
    train_data, test_data, val_data, angles = retrieve_data(folders_path)

    TrainSet = SHDataset(train_data, gaussians, folders_path)
    TestSet = SHDataset(test_data, gaussians, folders_path)
    ValSet = SHDataset(val_data, gaussians, folders_path)
    
    print(f'The Fitting Process May Commence')
    print(f'\nThe retrieved groups are {gaussians.get_groups().cpu().numpy()}')
    # model = SHMLP(gaussians.get_xyz.shape[0], scene.getTrainCameras().copy(), pipe, background, chkpt_path, gaussians.max_sh_degree, op.extract(args), gaussians.get_features_rest)
    model = SHsplitMLP(gaussians.get_xyz.shape[0], scene.getTrainCameras().copy(), pipe, background, chkpt_path, gaussians.max_sh_degree, op.extract(args), gaussians.get_features_rest, gaussians.get_groups(), gaussians)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    TrainLoader = DataLoader(TrainSet, batch_size=1, shuffle=True)
    # train_model(model, TrainLoader, optimizer)

def retrieve_data(path):
    '''
    Loading all the data (images and transformations) into one dictionary,
    splitting the data into training, testing and validation sets for each angle
    '''
    data_train = {}
    data_test = {}
    data_val = {}
    angles = []
    print(f'Loading the data from the dataset at {path}, giving the following folders {os.listdir(path)}')
    for folder in os.listdir(path):
        if not os.path.isdir(folder):
            pass
        angle = float(folder.replace(',', '.'))
        angles.append(angle)
        print(f'Preprocessing data for angle {angle}')
        folder_path = os.path.join(path, folder)
        data_train[angle] = {}
        data_test[angle] = {}
        data_val[angle] = {}

        with open(os.path.join(folder_path, 'transforms_train.json'), 'r') as file:
            content = json.load(file)
            frames = content['frames']
            for idx, frame in enumerate(frames):
                c2w = np.array(frame['transform_matrix'])
                c2w[:3, 1:3] *= -1
                w2c = np.linalg.inv(c2w)
                R = np.transpose(w2c[:3,:3])
                T = w2c[:3, 3]
                data_train[angle][idx] = {'image': frame['file_path'], 'R': R, 'T': T}

        with open(os.path.join(folder_path, 'transforms_test.json'), 'r') as file:
            content = json.load(file)
            frames = content['frames']
            for idx, frame in enumerate(frames):
                c2w = np.array(frame['transform_matrix'])
                c2w[:3, 1:3] *= -1
                w2c = np.linalg.inv(c2w)
                R = np.transpose(w2c[:3,:3])
                T = w2c[:3, 3]
                data_test[angle][idx] = {'image': frame['file_path'], 'R': R, 'T': T}
        
        with open(os.path.join(folder_path, 'transforms_val.json'), 'r') as file:
            content = json.load(file)
            frames = content['frames']
            for idx, frame in enumerate(frames):
                c2w = np.array(frame['transform_matrix'])
                c2w[:3, 1:3] *= -1
                w2c = np.linalg.inv(c2w)
                R = np.transpose(w2c[:3,:3])
                T = w2c[:3, 3]
                data_val[angle][idx] = {'image': frame['file_path'], 'R': R, 'T': T}
    
    return data_train, data_test, data_val, angles

def custom_loss(lambda_dssim, y_true, y_pred):
    Ll1 = l1_loss(y_pred, y_true)
    return (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim(y_pred, y_true))

def train_model(model, dataloader, optimizer, lambda_dssim = 0.2, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        loss_epoch = 0.0
        it = 0
        for sample, image in dataloader: # sample = (R, T, gamma_x, image_name) image_name format ./train/image_xxxx.png
            it += 1
            R, T, angle, image_name = sample
            optimizer.zero_grad()
            output = model(angle, image_name)
            loss = custom_loss(lambda_dssim, image, output)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
            if it > 1 :
                break
        print(f'Epoch {epoch} - Loss: {loss_epoch}')
    return None

class SHDataset(Dataset):
    def __init__(self, data, gaussians, path, transform=None):
        '''
        Arguments:
            data - list of tuples (angle, R, T, image)
            gaussians - the gaussian model containing (xyz, rotations, features)
            path - path to the directory containing the angle folders (with all the associated images for each angle)
            n_img - number of images
        '''
        self.path = path
        self.gaussians = gaussians
        self.n_img = len(data[0.0])
        print(f'DataSet containing {len(data)} angles and {self.n_img} images per angle')
        self.data = data
        self.transform = transform
        self.idx_map ={} # mapping idx_map[idx] = (angle, image) image ~ view position ~ (R, T, image)

        for angle_idx, angle in enumerate(data):
            for cam_idx, cam_data in enumerate(data[angle]):
                self.idx_map[angle_idx * self.n_img + cam_idx] = (angle_idx, cam_idx)

        self.angles = [angle for angle in data]

        print('SHDataset initialized')

    def __len__(self):
        return len(self.angles)

    def __getitem__(self, idx):
        '''
        Input: 
            - idx: one-dimensional indice
        Output:
            - sample: list [angle, R, T]
            - image: the ground truth image 
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        angle_idx, img_idx = self.idx_map[idx]
        angle_str = str(self.angles[angle_idx])
        if angle_str.endswith('.0'):
            angle_str = angle_str[:-2]
        img_name = os.path.join(self.path, angle_str.replace('.',','), self.data[self.angles[angle_idx]][img_idx]['image'][2:].replace('/', '\\') + '.png')
        print(f'Loading image {img_name}')
        image = read_image(img_name)
        # the line below got deported to the train_model method in order to save some memory transfer
        # xyz_rot, rotations_rot, features_rot = self.gaussians.external_rotation(3, np.radians(self.angles[angle_idx]), 'x')
        sample = [self.data[self.angles[angle_idx]][img_idx]['R'], self.data[self.angles[angle_idx]][img_idx]['T'], self.angles[angle_idx], self.data[self.angles[angle_idx]][img_idx]['image']]
        return sample, image
    
    @property
    def get_xyz(self):
        return self.gaussians.get_xyz
    
    @property
    def get_features(self):
        return self.gaussians.get_features

    def rotate_gaussians(self, idx, theta, axis='x'):
        '''
        Rotate the gaussians by theta degrees along the x-axis
        Input:
            idx - index of the gaussian group to rotate
            theta - angle in radians
            axis - axis to rotate along
        Output:
            rotated_xyz, rotated_rotations, rotated_features
        '''
        return self.gaussians.external_rotation(idx, theta, axis)

class DifferentiableRenderer(nn.Module):
    def __init__(self, chkpt, sh_degree, pipe, bg, opt):
        super(DifferentiableRenderer, self).__init__()
        '''
        Before saving the gaussians, they need to be grouped and pruned

        gaussians - the gaussian model containing the xyz, rotations and features
        features_rest - the original features_rest
        xyz - the original xyz coordinates of the gaussians
        groups - the groups of the gaussians
        '''
        self.gaussians = GaussianModel(sh_degree)
        if chkpt:
            (model_params, first_iter) = torch.load(chkpt)
            self.gaussians.restore(model_params, opt)
        
        self.xyz = self.gaussians.get_xyz
        self.rotations = self.gaussians.get_rots
        self.features_rest = self.gaussians.get_features_rest
        self.pipe = pipe
        self.bg = bg

    def forward(self, features_rest, viewpoint_cam, angle):
        self.gaussians.rotate_gaussians_MLP(3, angle, 'x', features_rest)
        rendered_image = self.renderer(viewpoint_cam)
        self.gaussins.set_features_rest(self.features_rest)
        self.gaussians.set_xyz(self.xyz)
        self.gaussians.set_rots(self.rotations)
        return rendered_image

    def renderer(self, viewpoint_cam):
        render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, self.bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        return read_image(image)

class SHMLP(nn.Module):
    def __init__(self, N, viewpoint_stack, pipe, bg, chkpt, sh_degree, opt, sh_coeff):
        super(SHMLP, self).__init__()
        self.N = N
        self.pipe = pipe
        self.bg = bg
        self.vewpoint_stack = viewpoint_stack

        # USING FULLY CONNECTED DENSE LAYERS

        self.sh_coeff = sh_coeff.view(self.N * 15 * 3, -1)

        self.input_dim_sh = self.N * 15 * 3
        self.input_dim_xyz = self.N * 3
        self.input_dim_angle = 1

        self.output_dim = self.N * 15 * 3

        O = 32
        A = 1024
        B = 1024
        C = 2048
        D = 2048
        E = 2048


        self.rotation_dense = nn.Linear(self.input_dim_angle, O ) # (1) -> (32)
        
        self.sh_dense = nn.Linear(self.input_dim_sh, A ) # (N * 15 * 3) -> A

        self.xyz_dense = nn.Linear(self.input_dim_xyz, B ) # (N * 3) -> B

        self.concat_dense_sh = nn.Linear(O + A , C) # (32 + A) -> C
        self.concat_dense_xyz = nn.Linear(O + B, D) # (32 + B) -> D

        self.concat_dense1 = nn.Linear(C + D , E) # (C + D) -> E

        self.output_dense = nn.Linear(E , self.output_dim) # (E) -> (N * 15 * 3)
        
        self.renderer = DifferentiableRenderer(chkpt, sh_degree, pipe, bg, opt)

        print('SHMLP Model initialized')
        print(self)
        print(f'Model size : {torch.cuda.memory_allocated()/1_000_000_000} Go')

    def forward(self, rotation_angle, image_name):
        '''
        Input:
            - rotation_angle: the angle of rotation of the viewpoint (in radians)
            - sh_coeff: the spherical harmonics coefficients of size (N, 15, 3) ~ features_rest
        '''
        rotation_angle = self.rotation_dense(rotation_angle)
        sh_coeff = self.sh_dense(self.sh_coeff)
        xyz = self.xyz_dense(self.xyz)
        concat_sh = torch.cat((rotation_angle, sh_coeff), dim=1)
        concat_xyz = torch.cat((rotation_angle, xyz), dim=1)
        concat = self.concat_dense1(torch.cat((concat_sh, concat_xyz), dim=1))
        output = self.output_dense(concat)

        # find the viewpoint_cam in the viewpoint_stack where the image_name matches
        image_name = os.path.basename(image_name)
        viewpoint_cam = [viewpoint for viewpoint in self.viewpoint_stack if viewpoint.image_name == image_name][0]
        rendered_image = self.renderer(output, viewpoint_cam, rotation_angle)
        
        return rendered_image

class SHsplitMLP(nn.Module):
    def __init__(self, N, viewpoint_stack, pipe, bg, chkpt, sh_degree, opt, sh_coeff, groups, gaussians):
        super(SHsplitMLP, self).__init__()
        '''
        Same MLP as SHMLP but with the sh_coeff split into 4 tensors of size (N_i, 15, 3)
        where the N_i represent the number of gaussians contained in each group.
        The model splits up the data spatially using precomputed groups thanks to the 
        bounding boxes of the gaussians. We will then treat each group independently and
        combine the results in order for each part to have access to the information
        contained in the other parts.
        '''
        self.N = N
        self.pipe = pipe
        self.bg = bg
        self.vewpoint_stack = viewpoint_stack
        self.groups = groups
        self.gaussians = GaussianModel(sh_degree)
        if chkpt:
            (model_params, first_iter) = torch.load(chkpt)
            self.gaussians.restore(model_params, opt)
        self.xyz = self.gaussians.get_xyz
        self.rotations = self.gaussians.get_rots
        self.features_rest = self.gaussians.get_features_rest

        # USING FULLY CONNECTED DENSE LAYERS IN A SPLITTED MANNER

        self.sh_coeff = sh_coeff.view(self.N, 15, 3)

        self.group1_mask = groups == 0
        self.group2_mask = groups == 1
        self.group3_mask = groups == 2
        self.group4_mask = groups == 3

        print(f'Mask for group 1 : {self.group1_mask}')

        self.N_1 = self.group1_mask.sum()
        self.N_2 = self.group2_mask.sum()
        self.N_3 = self.group3_mask.sum()
        self.N_4 = self.group4_mask.sum()

        print(f'Group 1 contains {self.N_1} gaussians')
        print(f'Group 2 contains {self.N_2} gaussians')
        print(f'Group 3 contains {self.N_3} gaussians')
        print(f'Group 4 contains {self.N_4} gaussians')

        self.sh_group1 = self.sh_coeff[self.group1_mask]
        self.sh_group2 = self.sh_coeff[self.group2_mask]
        self.sh_group3 = self.sh_coeff[self.group3_mask]
        self.sh_group4 = self.sh_coeff[self.group4_mask]

        self.xyz_group1 = self.xyz[self.group1_mask]
        self.xyz_group2 = self.xyz[self.group2_mask]
        self.xyz_group3 = self.xyz[self.group3_mask]
        self.xyz_group4 = self.xyz[self.group4_mask]

        A = 2048
        B = 2048
        C = 2048
        D = 2048
        E = 2048
        F = 2048
        G = 2048
        H = 2048
        I = 2048
        J = 2048
        K = 2048
        L = 2048
        M = 2048
        O = 2048
        P = 2048
        Q = 2048
        R = 2048
        S = 2048
        T = 2048
        U = 2048

        self.rotation_dense = nn.Linear(1, 32) # (1) -> (32)

        # Extract features from the spherical harmonics of each group
        self.sh_dense1_gr1 = nn.Linear(self.N_1 * 15 * 3, A) # (N_1 * 15 * 3) -> A
        self.sh_dense1_gr2 = nn.Linear(self.N_2 * 15 * 3, B) # (N_2 * 15 * 3) -> B
        self.sh_dense1_gr3 = nn.Linear(self.N_3 * 15 * 3, C) # (N_3 * 15 * 3) -> C
        self.sh_dense1_gr4 = nn.Linear(self.N_4 * 15 * 3, D) # (N_4 * 15 * 3) -> D

        # Extract features from the xyz coordinates of each group
        self.xyz_dense1_gr1 = nn.Linear(self.N_1 * 3, E) # (N_1 * 3) -> E
        self.xyz_dense1_gr2 = nn.Linear(self.N_2 * 3, F) # (N_2 * 3) -> F
        self.xyz_dense1_gr3 = nn.Linear(self.N_3 * 3, G) # (N_3 * 3) -> G
        self.xyz_dense1_gr4 = nn.Linear(self.N_4 * 3, H) # (N_4 * 3) -> H
        
        # Concatenate the features extracted from the spherical harmonics and the xyz coordinates
        self.sh_xyz_dense_gr12 = nn.Linear(32 + A + F, I) # (32 + A + F) -> I
        self.sh_xyz_dense_gr13 = nn.Linear(32 + A + G, J) # (32 + A + G) -> J
        self.sh_xyz_dense_gr14 = nn.Linear(32 + A + H, K) # (32 + A + H) -> K
        self.sh_xyz_dense_gr21 = nn.Linear(32 + B + E, L) 
        self.sh_xyz_dense_gr23 = nn.Linear(32 + B + G, M) # (32 + B + F) -> L
        self.sh_xyz_dense_gr24 = nn.Linear(32 + B + H, O) # (32 + B + H) -> M
        self.sh_xyz_dense_gr31 = nn.Linear(32 + C + E, P)
        self.sh_xyz_dense_gr32 = nn.Linear(32 + C + F, Q)
        self.sh_xyz_dense_gr34 = nn.Linear(32 + C + G, R) # (32 + C + G) -> O
        self.sh_xyz_dense_gr41 = nn.Linear(32 + D + E, S)
        self.sh_xyz_dense_gr42 = nn.Linear(32 + D + F, T)
        self.sh_xyz_dense_gr43 = nn.Linear(32 + D + G, U)

        self.output_gr1 = nn.Linear(I + J + K , self.N_1 * 15 * 3)
        self.output_gr2 = nn.Linear(L + M + O , self.N_2 * 15 * 3)
        self.output_gr3 = nn.Linear(P + Q , self.N_3 * 15 * 3)
        self.output_gr4 = nn.Linear(S + T + U , self.N_4 * 15 * 3)

        self.renderer = DifferentiableRenderer(chkpt, sh_degree, pipe, bg, opt)
        print('SHsplitMLP Model initialized')
        print(self)
        print(f'Model size : {torch.cuda.memory_allocated()/1_000_000_000} Go')

    def forward(self, rotation_angle, image_name):
        '''
        Input:
            - rotation_angle: the angle of rotation of the viewpoint (in radians)
            - sh_coeff: the spherical harmonics coefficients of size (N, 15, 3) ~ features_rest
        '''
        
        xyz, rotations, features = self.gaussians.external_rotation(3, rotation_angle, 'x')

        xyz1 = xyz[self.group1_mask]
        xyz2 = xyz[self.group2_mask]
        xyz3 = xyz[self.group3_mask]
        xyz4 = xyz[self.group4_mask]

        rotation_angle = self.rotation_dense(rotation_angle)
        
        sh_group1 = self.sh_dense1_gr1(self.sh_group1)
        sh_group2 = self.sh_dense1_gr2(self.sh_group2)
        sh_group3 = self.sh_dense1_gr3(self.sh_group3)
        sh_group4 = self.sh_dense1_gr4(self.sh_group4)

        xyz_group1 = self.xyz_dense1_gr1(xyz1)
        xyz_group2 = self.xyz_dense1_gr2(xyz2)
        xyz_group3 = self.xyz_dense1_gr3(xyz3)
        xyz_group4 = self.xyz_dense1_gr4(xyz4)

        concat_sh_xyz_gr12 = self.sh_xyz_dense_gr12(torch.cat((rotation_angle, sh_group1, xyz_group2), dim=1))
        concat_sh_xyz_gr13 = self.sh_xyz_dense_gr13(torch.cat((rotation_angle, sh_group1, xyz_group3), dim=1))
        concat_sh_xyz_gr14 = self.sh_xyz_dense_gr14(torch.cat((rotation_angle, sh_group1, xyz_group4), dim=1))
        concat_sh_xyz_gr21 = self.sh_xyz_dense_gr21(torch.cat((rotation_angle, sh_group2, xyz_group1), dim=1))
        concat_sh_xyz_gr23 = self.sh_xyz_dense_gr23(torch.cat((rotation_angle, sh_group2, xyz_group3), dim=1))
        concat_sh_xyz_gr24 = self.sh_xyz_dense_gr24(torch.cat((rotation_angle, sh_group2, xyz_group4), dim=1))
        concat_sh_xyz_gr31 = self.sh_xyz_dense_gr31(torch.cat((rotation_angle, sh_group3, xyz_group1), dim=1))
        concat_sh_xyz_gr32 = self.sh_xyz_dense_gr32(torch.cat((rotation_angle, sh_group3, xyz_group2), dim=1))
        concat_sh_xyz_gr34 = self.sh_xyz_dense_gr34(torch.cat((rotation_angle, sh_group3, xyz_group4), dim=1))
        concat_sh_xyz_gr41 = self.sh_xyz_dense_gr41(torch.cat((rotation_angle, sh_group4, xyz_group1), dim=1))
        concat_sh_xyz_gr42 = self.sh_xyz_dense_gr42(torch.cat((rotation_angle, sh_group4, xyz_group2), dim=1))
        concat_sh_xyz_gr43 = self.sh_xyz_dense_gr43(torch.cat((rotation_angle, sh_group4, xyz_group3), dim=1))
                                                    
        output_gr1 = self.output_gr1(torch.cat((concat_sh_xyz_gr12, concat_sh_xyz_gr13, concat_sh_xyz_gr14), dim=1))
        output_gr2 = self.output_gr2(torch.cat((concat_sh_xyz_gr21, concat_sh_xyz_gr23, concat_sh_xyz_gr24), dim=1))
        output_gr3 = self.output_gr3(torch.cat((concat_sh_xyz_gr31, concat_sh_xyz_gr32, concat_sh_xyz_gr34), dim=1))
        output_gr4 = self.output_gr4(torch.cat((concat_sh_xyz_gr41, concat_sh_xyz_gr42, concat_sh_xyz_gr43), dim=1))

        output = torch.cat((output_gr1, output_gr2, output_gr3, output_gr4), dim=0)

        # find the viewpoint_cam in the viewpoint_stack where the image_name matches
        image_name = os.path.basename(image_name)
        viewpoint_cam = [viewpoint for viewpoint in self.viewpoint_stack if viewpoint.image_name == image_name][0]
        rendered_image = self.renderer(output, viewpoint_cam, rotation_angle)
        
        return rendered_image

class SHConv(nn.Module):
    def __init__(self, N, viewpoint_stack, pipe, bg, chkpt, sh_degree, opt, sh_coeff):
        super(SHConv, self).__init__()
        '''
        
        if convolutional layers are used, we will keep the original dimensions of the spherical
        harmonics (N, 15, 3). We will optimize the process by spatially reordering the gaussians,
        and therefore the xyz, features and rotations tensors depending on the gaussians' positions in space
        '''
        self.N = N
        self.pipe = pipe
        self.bg = bg
        self.vewpoint_stack = viewpoint_stack

        self.sh_coeff = sh_coeff.view(self.N, 15, 3)
        
        # USING CONVOLUTIONAL LAYERS

        self.sh_coeff = sh_coeff.view(self.N, 15, 3)

        self.input_dim_sh = (self.N, 15, 3)
        self.input_dim_xyz = (self.N, 3)
        self.input_dim_angle = 1

        self.rotation_conv = nn.Conv1d(1, 32, 1) # (1) -> (32)

        print('SHConv Model initialized')
        print(self)
        print(f'Model size : {torch.cuda.memory_allocated()/1_000_000_000} Go')

    def forward(self, rotation_angle, image_name):
        '''
        Input:
            - rotation_angle: the angle of rotation of the viewpoint (in radians)
            - sh_coeff: the spherical harmonics coefficients of size (N, 15, 3) ~ features_rest
        '''
        image_name = os.path.basename(image_name)
        viewpoint_cam = [viewpoint for viewpoint in self.viewpoint_stack if viewpoint.image_name == image_name][0]
        rendered_image = self.renderer(output, viewpoint_cam, rotation_angle)
        
        return rendered_image
#################################################################################################
#################################################################################################

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--json_bounds", type=str, default = None)
    parser.add_argument("--sh_fitting", type=str, default = None)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    dataset, gaussians, scene, pipe, background = training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.json_bounds, args.sh_fitting)
    # All done
    print("\nTraining complete.")

    shs_fit(args, dataset, gaussians, scene, pipe, background)