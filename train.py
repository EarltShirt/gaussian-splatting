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
from IPython.display import display, HTML
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
from PIL import Image, ImageDraw

from segment_anything import sam_model_registry, SamPredictor

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_mask(mask, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = np.zeros((h, w, 4), dtype=np.uint8)
    mask_image[:, :, :3] = (mask.reshape(h, w, 1) * color[:3] * 255).astype(np.uint8)
    mask_image[:, :, 3] = (mask.reshape(h, w) * color[3] * 255).astype(np.uint8)
    return Image.fromarray(mask_image, 'RGBA')

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


class InteractiveSegmenter:

    def __init__(self, image, predictor):
        self.image = image
        self.predictor = predictor
        self.input_points = []
        self.input_labels = []
        self.input_box = []
        self.colors = [np.random.random(4) for _ in range(10)]
        self.num_green_points = 1
        self.num_red_points = 0
        self.num_boxes = 0
        self.fig = None
        self.ax = None
        self.cid = None

    def ask_for_number_of_points(self):
        print("How many green points would you like to add?")
        self.num_green_points = int(input())
        print("How many red points would you like to add?")
        self.num_red_points = int(input())
        # print("How many boxes would you like to add? (0 or 1 box)")
        # self.num_boxes = int(input())
        # assert self.num_boxes in [0, 1], "Only 0 or 1 box is supported for now."

    def retrieve_all(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.imshow(self.image)
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        plt.axis('on')
        plt.show()


    def onclick(self, event):
        ix, iy = event.xdata, event.ydata
        if ix is not None and iy is not None:
            if len(self.input_points) < self.num_green_points:
                self.input_points.append([ix, iy])
                self.input_labels.append(1)  # Assigning label 1 for green points
            elif len(self.input_points) < self.num_green_points + self.num_red_points:
                self.input_points.append([ix, iy])
                self.input_labels.append(0)  # Assigning label 0 for red points
            elif len(self.input_box) < 4:
                self.input_box += [ix, iy]
                self.ax.scatter(ix, iy, marker='o', s=375, color='blue', linewidth=1.25) 
                # print("Box :", self.input_box)
            self.ax.clear()  # Clear the current plot
            self.ax.imshow(self.image)  # Replot the image
            show_points(np.array(self.input_points), np.array(self.input_labels), self.ax)  # Plot all points

            if len(self.input_box) == 4:
                show_box(self.input_box, self.ax)
            plt.draw()
            
            if (len(self.input_points) == self.num_green_points + self.num_red_points) and (len(self.input_box) == 4 * self.num_boxes):
                self.fig.canvas.mpl_disconnect(self.cid)
                self.compute_masks()
        plt.draw()  # Update the plot

    def compute_masks(self):
        input_points_array = np.array(self.input_points)
        input_labels_array = np.array(self.input_labels)
        input_box_array = np.array(self.input_box)
        print("Box : ", input_box_array)
        
        if len(input_box_array) != 0:
            masks, scores, logits = self.predictor.predict(
                point_coords=input_points_array,
                point_labels=input_labels_array,
                box=input_box_array[None, :],
                multimask_output=True,
            )
        else:
            masks, scores, logits = self.predictor.predict(
                point_coords=input_points_array,
                point_labels=input_labels_array,
                multimask_output=True,
            )
        
        num_masks = len(masks)
        rows = 1
        cols = num_masks

        fig, axes = plt.subplots(rows, cols, figsize=(15, 7))

        for i, (mask, score) in enumerate(zip(masks, scores)):
            color = self.colors[i % len(self.colors)]
            mask_rgba = show_mask(mask, random_color=True)
            # self.gaussians.add_2Dmask(mask_rgba, i)
            
            ax = axes[i] if num_masks > 1 else axes
            ax.imshow(self.image)
            ax.imshow(mask_rgba)
            ax.set_title(f"Mask {i + 1} (score: {score:.2f})")
            show_points(input_points_array, input_labels_array, ax)
            ax.axis('off')

        plt.tight_layout()
        plt.show()

def segment_anything(image): # image is given in float32, which isn't supported by PIL
    print("Reading image...")
    image = np.transpose(image, (1, 2, 0))
    print(f"Value example in image : {image[500,500,:]}")
    image = (image * 255).astype(np.uint8)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("Loading model...")
    sam_checkpoint = "./sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    print("Loading model on device...")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    # interactive_segmenter = InteractiveSegmenter(image, predictor, gaussians)
    interactive_segmenter = InteractiveSegmenter(image, predictor)
    interactive_segmenter.ask_for_number_of_points()
    interactive_segmenter.retrieve_all()

# def __init__(self, image, predictor, gaussians):
#     self.image = image
#     self.predictor = predictor
#     self.input_points = []
#     self.input_labels = []
#     self.input_box = []
#     self.colors = [np.random.random(4) for _ in range(10)]
#     self.num_green_points = 1
#     self.num_red_points = 0
#     self.num_boxes = 0
#     self.fig = None
#     self.ax = None
#     self.cid = None
#     self.gaussians = gaussians

# def segment_anything(gaussians, image, part, view): # image is given in float32, which isn't supported by PIL
#     print("Reading image...")
#     image = np.transpose(image, (1, 2, 0))
#     print(f"Value example in image : {image[500,500,:]}")
#     image = (image * 255).astype(np.uint8)
#     # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     print("Loading model...")
#     sam_checkpoint = "./sam_vit_h_4b8939.pth"
#     model_type = "vit_h"
#     device = "cuda"
#     print("Loading model on device...")
#     sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
#     sam.to(device=device)
#     predictor = SamPredictor(sam)
#     predictor.set_image(image)
#     interactive_segmenter = InteractiveSegmenter(image, predictor, gaussians)
#     interactive_segmenter.ask_for_number_of_points()
#     interactive_segmenter.retrieve_all(part, view)

# def retrieve_all(self, part, view):
#     self.fig, self.ax = plt.subplots(figsize=(10, 10))
#     self.ax.imshow(self.image)
#     self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick(part, view))
#     plt.axis('on')
#     plt.show()

# def onclick(self, event, part, view):
#     ix, iy = event.xdata, event.ydata
#     if ix is not None and iy is not None:
#         if len(self.input_points) < self.num_green_points:
#             self.input_points.append([ix, iy])
#             self.input_labels.append(1)  # Assigning label 1 for green points
#         elif len(self.input_points) < self.num_green_points + self.num_red_points:
#             self.input_points.append([ix, iy])
#             self.input_labels.append(0)  # Assigning label 0 for red points
#         elif len(self.input_box) < 4:
#             self.input_box += [ix, iy]
#             self.ax.scatter(ix, iy, marker='o', s=375, color='blue', linewidth=1.25) 
#             # print("Box :", self.input_box)
#         self.ax.clear()  # Clear the current plot
#         self.ax.imshow(self.image)  # Replot the image
#         show_points(np.array(self.input_points), np.array(self.input_labels), self.ax)  # Plot all points
#         if len(self.input_box) == 4:
#             show_box(self.input_box, self.ax)
#         plt.draw()
#         if (len(self.input_points) == self.num_green_points + self.num_red_points) and (len(self.input_box) == 4 * self.num_boxes):
#             self.fig.canvas.mpl_disconnect(self.cid)
#             self.compute_masks(part, view)
#     plt.draw()  # Update the plot

# def compute_masks(self, part, view):
#     input_points_array = np.array(self.input_points)
#     input_labels_array = np.array(self.input_labels)
#     input_box_array = np.array(self.input_box)
#     print("Box : ", input_box_array)
#     if len(input_box_array) != 0:
#         masks, scores, logits = self.predictor.predict(
#             point_coords=input_points_array,
#             point_labels=input_labels_array,
#             box=input_box_array[None, :],
#             multimask_output=True,
#         )
#     else:
#         masks, scores, logits = self.predictor.predict(
#             point_coords=input_points_array,
#             point_labels=input_labels_array,
#             multimask_output=True,
#         )
#     num_masks = len(masks)
#     rows = 1
#     cols = num_masks
#     fig, axes = plt.subplots(rows, cols, figsize=(15, 7))
#     for i, (mask, score) in enumerate(zip(masks, scores)):
#         color = self.colors[i % len(self.colors)]
#         mask_rgba = show_mask(mask, random_color=True)
#         self.gaussians.add_2Dmask(mask_rgba, part, view)
#         ax = axes[i] if num_masks > 1 else axes
#         ax.imshow(self.image)
#         ax.imshow(mask_rgba)
#         ax.set_title(f"Mask {i + 1} (score: {score:.2f})")
#         show_points(input_points_array, input_labels_array, ax)
#         ax.axis('off')
#     plt.tight_layout()
#     plt.show()
#################################################################################################
#################################################################################################

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
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

        #################################################################################################
        ############################################ SEGMENTATION #######################################
        if iteration == 10000 :
            # Convert image to numpy array
            # print(f"dir(image) : {dir(image)}")
            seg_viewpoint_stack = scene.getTrainCameras().copy()
            seg_viewpoint_cam = seg_viewpoint_stack.pop(20)
            seg_render_pkg = render(seg_viewpoint_cam, gaussians, pipe, bg)
            seg_image = seg_render_pkg["render"]
            seg_image = seg_image.detach().cpu().numpy()
            segment_anything(seg_image)

        # if iteration == 20000 :
        #     seg_viewpoint_stack = scene.getTrainCameras().copy()
        #     for part in range(4) : # there are 4 main parts to be segmented on the UR
        #         for view in range(len(seg_viewpoint_stack)) :
        #             seg_viewpoint_stack = scene.getTrainCameras().copy()
        #             seg_viewpoint_cam = seg_viewpoint_stack.pop(view)
        #             seg_render_pkg = render(seg_viewpoint_cam, gaussians, pipe, bg)
        #             seg_image = seg_render_pkg["render"]
        #             seg_image = seg_image.detach().cpu().numpy()
        #             segment_anything(gaussians, seg_image, part, view)
        #################################################################################################
        #################################################################################################   
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
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                print("Saving {} Gaussians".format(scene.gaussians.get_xyz.shape[0]))
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
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

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
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
