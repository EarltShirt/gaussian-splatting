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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def load_bounds(file_path):
    # load the bounds dictionary from the json file
    with open(file_path, 'r') as file:
        bounds = json.load(file)
    return bounds


#################################################################################################
#################################################################################################


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, bounds_file):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
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
            # if iteration % 500 == 0 and iteration > 2000 and iteration < 10000:
            #     print("\n[ITER {}] Regrouping Gaussians".format(iteration))
            #     scene.gaussians.regroup_and_prune()
            
            if iteration == 8000:
                print("\n[ITER {}] Group Visualization".format(iteration))
                segmented_ply_path = os.path.join(scene.model_path, "segmented.ply")    
                gaussians.save_segmented_ply(segmented_ply_path)

                # gaussians.regroup_and_prune()
                post_segmented_ply_path = os.path.join(scene.model_path, "post_segmented.ply")
                gaussians.save_post_segmented_ply(post_segmented_ply_path)

                print("\nStoring the pre-segmented point cloud at {}".format(segmented_ply_path))
                print("\nStoring the post-segmented point cloud at {}".format(post_segmented_ply_path))
                
                # theta = -np.pi / 4
                # ROT = [
                #     [1, 0, 0], 
                #     [0, np.cos(theta), np.sin(theta)], 
                #     [0, -np.sin(theta), np.cos(theta)] ]
                # rotation_tensor = torch.tensor(ROT, dtype=torch.float, device="cpu")
                # rotated_ply_path = os.path.join(scene.model_path, "rotated.ply")
                # gaussians.store_rotated_groups(rotated_ply_path, 3, rotation_tensor)
                # print("\nStoring the rotated point cloud at {}".format(rotated_ply_path))

            if iteration == 25501:
                gaussians.regroup_and_prune()
                theta = - np.pi / 4
                # rotation_tensor = torch.tensor(ROT, dtype=torch.float, device="cuda")
                gaussians.rotate_gaussians(3, theta, axis='x')

            if iteration == 25502:
                print("\n[ITER {}] Pausing the training for the user to check the results".format(iteration))
                input("Press Enter to continue...")
    
    # After the training is done, we start training the SH coefficients
    shs_fit(dataset, gaussians, pipe, scene)

def preprocess_data(path):
    '''
    The associated transformation matrix can be found 
    in the associated transformation.json file
    '''
    data = {}
    data['train'] = {}
    data['test'] = {}
    for folder in os.listdir(path):
        angle = float(folder)
        folder_path = os.path.join(path, folder)
        files = os.listdir(folder_path)
        train_files = files[:int(0.8 * len(files))]
        test_files = files[int(0.8 * len(files)):]
        data['train'][angle] = [os.path.join(folder_path, file) for file in train_files]
        data['test'][angle] = [os.path.join(folder_path, file) for file in test_files]
    return data

def custom_loss(opt, y_true, y_pred):
    Ll1 = l1_loss(y_pred, y_true)
    return (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(y_pred, y_true))

def train_model(model, dataloader, optimizer, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for rotation_angles, ground_truth_images in dataloader:
            optimizer.zero_grad()
            outputs = model(rotation_angles, sh_coefficients)
            loss = custom_loss(ground_truth_images, outputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}')

def shs_fit(path, opt, pipe, scene, gaussians, bg, data):
    data = preprocess_data(path)
    N = gaussians.get_xyz.shape[0]
    num_samples = ... # Number of different angles
    batch_size = 32
    width, height = 1000, 1000
    train_model(model, data, opt)

    rotation_angles = torch.randn(num_samples, 1) 
    sh_coefficients = torch.randn(N, 16, 3)
    ground_truth_images = torch.randn(num_samples, 3, width, height)

    dataset = TensorDataset(rotation_angles, sh_coefficients, ground_truth_images)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SHModel(num_samples)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, dataloader, optimizer)

class DifferentiableRenderer(nn.Module):
    def __init__(self, gaussians, viewpoint_cam, pipe, bg):
        super(DifferentiableRenderer, self).__init__()
        self.gaussians = gaussians
        self.viewpoint_cam = viewpoint_cam
        self.pipe = pipe
        self.bg = bg

    def forward(self, sh_coeff):
        self.gaussians.set_features_rest(sh_coeff)
        rendered_image = self.renderer(self.viewpoint_cam, self.gaussians, self.pipe, self.bg)
        return rendered_image

    def renderer(self, viewpoint_cam, gaussians, pipe, bg):
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        return image
# Define the neural network model
class SHModel(nn.Module):
    def __init__(self, N):
        super(SHModel, self).__init__()
        self.N = N
        
        self.rotation_dense1 = nn.Linear(1, 64)
        self.rotation_dense2 = nn.Linear(64, 64)
        
        self.sh_dense1 = nn.Linear(N * 16 * 3, 512)
        self.sh_dense2 = nn.Linear(512, 512)
        
        self.concat_dense1 = nn.Linear(512 + 64, 512)
        self.concat_dense2 = nn.Linear(512, 512)
        
        self.output_dense = nn.Linear(512, N * 16 * 3)
        
        self.renderer = DifferentiableRenderer()
    
    def forward(self, rotation_angle, sh_coeff):
        sh_coeff_flat = sh_coeff.view(sh_coeff.size(0), -1)
        
        x1 = F.relu(self.rotation_dense1(rotation_angle))
        x1 = F.relu(self.rotation_dense2(x1))
        
        x2 = F.relu(self.sh_dense1(sh_coeff_flat))
        x2 = F.relu(self.sh_dense2(x2))
        
        x = torch.cat((x1, x2), dim=1)
        
        x = F.relu(self.concat_dense1(x))
        x = F.relu(self.concat_dense2(x))
        
        adjusted_sh_coeff_flat = self.output_dense(x)
        adjusted_sh_coeff = adjusted_sh_coeff_flat.view(-1, self.N, 16, 3)
        
        rendered_image = self.renderer(adjusted_sh_coeff)
        
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
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.json_bounds)

    # All done
    print("\nTraining complete.")


