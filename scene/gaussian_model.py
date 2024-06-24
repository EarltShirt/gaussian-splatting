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

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH, SH2RGB
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

from scipy.spatial.transform import Rotation as R
import quaternion
from e3nn import o3
import einops

# I need this function in order to store the post-segmented point cloud
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

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        # list containing the group index for each gaussian (gaussians 
        # are supposed to stay in the same order in the _xyz tensor)
        self._groups = torch.empty(0)
        self.bounds = None
        self.parts = None
        self.x_pivots = None
        self.z_pivots = None


    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
#################################################################################################
####################################### MY ADDITIONS ############################################
    def set_groups(self, np_groups):
        self._groups = torch.tensor(np.asarray(np_groups)).float().cuda()

    def set_group(self, gaussian_idx, group_idx):
        self._groups[gaussian_idx] = group_idx

    def set_group(self, gaussian_idx_start, gaussian_idx_end, group_idx):
        self._groups[gaussian_idx_start:gaussian_idx_end] = group_idx

    def get_group(self, gaussian_idx):
        return self._groups[gaussian_idx]
    
    def get_groups(self):
        return self._groups

    def set_bounds(self, bounds):
        self.bounds = bounds

    def set_parts(self, parts):
        self.parts = parts

    def get_pivots(self):
        return self.pivots
    
    def get_x_pivot(self, part_idx):
        switcher = {
            1: 'part_1',
            2: 'part_2',
            3: 'part_3',
            4: 'part_4'
        }
        return self.x_pivots[switcher.get(part_idx, "Invalid part index")]
    
    def get_z_pivot(self, part_idx):
        switcher = {
            1: 'part_1',
            2: 'part_2',
            3: 'part_3',
            4: 'part_4'
        }
        print(f"Calling get_pivot with part_idx : {part_idx}")
        return self.z_pivots[switcher.get(part_idx, "Invalid part index")]

    def define_pivots(self):
        x_pivots = {}
        z_pivots = {}
        for group_idx, bound in self.bounds.items():
            if 'part1_angle_l' in group_idx:
                x_pivots['part_1'] = (np.array(bound["min"]) + np.array(bound["max"]))/2
            elif 'part2_angle_l' in group_idx:
                x_pivots['part_2'] = (np.array(bound["min"]) + np.array(bound["max"]))/2
            elif 'part3_angle_l' in group_idx:
                x_pivots['part_3'] = (np.array(bound["min"]) + np.array(bound["max"]))/2
            elif 'part4_angle_l' in group_idx:
                x_pivots['part_4'] = (np.array(bound["min"]) + np.array(bound["max"]))/2
            
            elif 'part1_angle_h' in group_idx:
                z_pivots['part_1'] = (np.array(bound["min"]) + np.array(bound["max"]))/2
            elif 'part2_angle_h' in group_idx:
                z_pivots['part_2'] = (np.array(bound["min"]) + np.array(bound["max"]))/2
            elif 'part3_angle_h' in group_idx:
                z_pivots['part_3'] = (np.array(bound["min"]) + np.array(bound["max"]))/2
            elif 'part4_angle_h' in group_idx:
                z_pivots['part_4'] = (np.array(bound["min"]) + np.array(bound["max"]))/2
        
        self.x_pivots = x_pivots
        self.z_pivots = z_pivots

    def rotate_group(self, group_idx : int, rotation : torch.Tensor):
        '''
        Rotate all gaussians in the group group_idx by the rotation 
        matrix rotation around the pivot point pivot_point
        Every Gaussian of the higher level groups have also to be rotated
        
        Args:
            rotation : torch.Tensor of shape (3, 3)
            pivot_point : torch.Tensor of shape (3,)
        Returns:
            None, the gaussians are rotated in place
        '''
        pivot_point = self.get_x_pivot(group_idx)
        group_mask = self._groups >= group_idx
        self._xyz[group_mask] = torch.bmm(rotation, (self._xyz[group_mask] - pivot_point).unsqueeze(-1)).squeeze(-1) + pivot_point

    def create_rotation_matrix(self, theta, axis):
        '''
        Create a rotation matrix around the axis axis by the given angle (in radians)
        '''
        if axis == 'x':
            ROT = [
                [1, 0, 0], 
                [0, np.cos(theta), np.sin(theta)], 
                [0, -np.sin(theta), np.cos(theta)] ]
        elif axis == 'z':
            ROT = [
                [np.cos(theta), np.sin(theta), 0], 
                [-np.sin(theta), np.cos(theta), 0], 
                [0, 0, 1] ]
        return torch.tensor(ROT, dtype=torch.float32, device="cuda")
    
    def create_rotation_quaternion(self, theta, axis):
        '''
        Create a rotation quaternion around the axis axis by the given angle (in radians)
        '''
        if axis == 'x':
            ROT = [np.cos(theta/2), np.sin(theta/2), 0, 0]
        elif axis == 'z':
            ROT = [np.cos(theta/2), 0, 0, np.sin(theta/2)]
        return torch.tensor(ROT, dtype=torch.float32, device="cpu")

    def rotate_groups_test(self, group_idx : int, rotation : torch.Tensor):
        '''
        Same as before, but not in place in order to test the function
        
        Args:
            rotation : torch.Tensor of shape (3, 3)
            pivot_point : torch.Tensor of shape (3,)
        Returns:
            New xyz tensor
        '''
        pivot_point = torch.tensor(self.get_x_pivot(group_idx), dtype=torch.float32, device="cpu")
        xyz = self._xyz.clone().cpu().float()
        group_mask = (self._groups >= group_idx).cpu()
        N = int(group_mask.sum().item())
        rotations = rotation.repeat(N, 1, 1).float()
        xyz[group_mask] = torch.add(torch.bmm(rotations, (xyz[group_mask] - pivot_point).unsqueeze(-1)).squeeze(-1).cpu(), pivot_point.repeat(N, 1).cpu())
        return xyz, group_mask

    def store_rotated_groups(self, path : str, group_idx : int, rotation : torch.Tensor):
        self.define_pivots()
        rotated_xyz, rotated_mask = self.rotate_groups_test(group_idx, rotation)
        colors = np.zeros((rotated_xyz.shape[0], 3))
        colors[rotated_mask] = np.array([128, 0, 128])
        storePly(path, rotated_xyz.detach().cpu().numpy(), colors)

    # method found inn the issues of the inria/3DGS github repo
    # def transform_shs(self, shs_feat, rotation_matrix):
    def transform_shs(self, shs_feat, theta, axis):
        # rotate shs
        # P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]) # switch axes: yzx -> xyz
        # P = np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]])
        # permuted_rotation_matrix = np.linalg.inv(P) @ rotation_matrix @ P
        # rot_angles = o3._rotation.matrix_to_angles(torch.from_numpy(permuted_rotation_matrix))
        rotation_matrix = self.create_rotation_matrix(theta, axis).cpu().numpy()
        rot_angles = o3._rotation.matrix_to_angles(torch.from_numpy(rotation_matrix))
        D_1 = o3.wigner_D(1, rot_angles[0], - rot_angles[1], rot_angles[2]).float()
        D_2 = o3.wigner_D(2, rot_angles[0], - rot_angles[1], rot_angles[2]).float()
        D_3 = o3.wigner_D(3, rot_angles[0], - rot_angles[1], rot_angles[2]).float()

        # Using custom quaternion construction
        # rot_angles = self.create_rotation_quaternion(theta, axis)
        # # Construction coefficient
        # D_1 = o3.wigner_D(1, rot_angles[0], rot_angles[1], rot_angles[2]).float()
        # D_2 = o3.wigner_D(2, rot_angles[0], rot_angles[1], rot_angles[2]).float()
        # D_3 = o3.wigner_D(3, rot_angles[0], rot_angles[1], rot_angles[2]).float()

        #rotation of the shs features
        one_degree_shs = shs_feat[:, 0:3]
        one_degree_shs = einops.rearrange(one_degree_shs, 'n shs_num rgb -> n rgb shs_num')
        one_degree_shs = einops.einsum(
                D_1,
                one_degree_shs.cpu().float(),
                "... i j, ... j -> ... i",
            )
        one_degree_shs = einops.rearrange(one_degree_shs, 'n rgb shs_num -> n shs_num rgb')
        shs_feat[:, 0:3] = one_degree_shs

        two_degree_shs = shs_feat[:, 3:8]
        two_degree_shs = einops.rearrange(two_degree_shs, 'n shs_num rgb -> n rgb shs_num')
        two_degree_shs = einops.einsum(
                D_2,
                two_degree_shs.cpu().float(),
                "... i j, ... j -> ... i",
            )
        two_degree_shs = einops.rearrange(two_degree_shs, 'n rgb shs_num -> n shs_num rgb')
        shs_feat[:, 3:8] = two_degree_shs

        three_degree_shs = shs_feat[:, 8:15]
        three_degree_shs = einops.rearrange(three_degree_shs, 'n shs_num rgb -> n rgb shs_num')
        three_degree_shs = einops.einsum(
                D_3,
                three_degree_shs.cpu().float(),
                "... i j, ... j -> ... i",
            )
        three_degree_shs = einops.rearrange(three_degree_shs, 'n rgb shs_num -> n shs_num rgb')
        shs_feat[:, 8:15] = three_degree_shs

        return shs_feat

    def rotate_gaussians(self, group_idx : int, theta : float, axis : str):
        self.regroup_and_prune()
        self.define_pivots()
        pivot_point = torch.tensor(self.get_x_pivot(group_idx), dtype=torch.float, device="cuda")
        group_mask = self._groups >= group_idx
        N = int(group_mask.sum().item())
        rotation = self.create_rotation_matrix(theta, axis)
        rotations = rotation.repeat(N, 1, 1).float()
        bmm = torch.bmm(rotations, (self._xyz[group_mask] - pivot_point).unsqueeze(-1)).squeeze(-1)
        repeated_pivot = pivot_point.repeat(N, 1).float()
        self._xyz[group_mask] = torch.add(bmm, repeated_pivot).float()

        
        test = self._rotation[100:110].clone()
        r = build_rotation(test).double()
        rot_angles = o3._rotation.matrix_to_quaternion(r.cpu())
        print(f'\n\nQuaternions before using o3 :\n{test.double().cpu().numpy()}')
        print(f'Quaternions after using o3  :\n{rot_angles.double().cpu().numpy()}')
        print(f'Difference between the two methods : \n{(rot_angles - test.cpu()).cpu().numpy()}')
        print(f'Norm of the difference : {torch.norm(rot_angles - test.cpu())}')
        print(f'Difference between the two methods with 3/4: \n{(rot_angles*3/4 - test.cpu()).cpu().numpy()}')
        print(f'Norm of the difference with 3/4: {torch.norm(rot_angles*3/4 - test.cpu())}')
        
        
        # same but for the internal rotation of the gaussians
        rotations = rotation.repeat(N, 1, 1).double()
        rotated_rotations = build_rotation(self._rotation[group_mask]).double()
        rotated_rotations = torch.bmm(rotations, rotated_rotations)
        angles = o3._rotation.matrix_to_quaternion(rotated_rotations.cpu()).float()
        self._rotation[group_mask] = angles.to(device="cuda")
        
        # Now we will rotate the features_dc and features_rest
        # print(f'shape of features_rest : {self._features_rest.shape}')
        # theta = -np.pi / 4
        # ROT = self.create_rotation_matrix(theta, 'z')
        # self._features_rest[group_mask] = self.transform_shs(self.get_features[group_mask][:,:15,:], ROT)
        self._features_rest[group_mask] = self.transform_shs(self.get_features[group_mask][:,:15,:], theta, axis)

    def prune_points_groups(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        self._groups = self._prune_optimizer_groups(valid_points_mask)

    def _prune_optimizer_groups(self, mask):
        groups_tensor = self._groups[mask]
        return groups_tensor

    def cat_tensors_to_optimizer_groups(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    

    def densification_postfix_groups(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_groups):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer_groups(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # only concatenation when cloning, use of prune_points_groups after the splitting
        self._groups = torch.cat((self._groups, new_groups))

    def densify_and_split_groups(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        # standard deviation for the new gaussians
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        # means for the new gaussians
        means =torch.zeros((stds.size(0), 3),device="cuda")
        # generate N samples for each new gaussian
        samples = torch.normal(mean=means, std=stds)
        # Build the rotations
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        # apply the rotations to the new gaussians an translate them to the original position
        # this isn't a concatenation, it's a element-wise sum
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        # adjust the scaling for the new gaussians
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        # replicate the other attributes of the splitted gaussians
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        new_groups = self._groups[selected_pts_mask].repeat(N, 1).flatten()

        # concatenate the new gaussians to the existing ones
        self.densification_postfix_groups(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_groups)
        # create mask to prune original points that were split
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        # prune the points
        self.prune_points_groups(prune_filter)

    def densify_and_clone_groups(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_groups = self._groups[selected_pts_mask]

        self.densification_postfix_groups(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_groups)

    def densify_and_prune_groups(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone_groups(grads, max_grad, extent)
        self.densify_and_split_groups(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points_groups(prune_mask)

        torch.cuda.empty_cache()

    def regroup_gaussians(self):
        '''
        Reassings the groups to the gaussians based on their position using the
        associated bounds (group_idxc in [1, #groups], -1 for environment, 0 for no group)
        '''

        epsilon = np.ones(3) * 0.001
        groups = torch.zeros((self._xyz.shape[0], ), device="cuda")
        xyz = self._xyz.detach().cpu().numpy()

        for group_idx, bound in self.bounds.items():
            if 'part1' in group_idx:
                idx = 1
            elif 'part2' in group_idx:
                idx = 2
            elif 'part3' in group_idx:
                idx = 3
            elif 'part4' in group_idx:
                idx = 4
            else :
                idx = 0
            group_mask = torch.where(
                torch.all(self._xyz >= torch.from_numpy(np.array(bound["min"])).to(device="cuda"), dim=1), True, False
            )
            group_mask = torch.logical_and(group_mask, torch.all(self._xyz <= torch.from_numpy(np.array(bound["max"])).to(device="cuda"), dim=1))
            groups[group_mask] = idx
        self._groups = groups

    def regroup_and_prune(self):
        '''
        Removes the points that are not assigned to any group
        '''
        n_gaussians = self.get_xyz.shape[0]
        self.regroup_gaussians()
        group_mask = self._groups == 0
        self.prune_points_groups(group_mask)
        print(f'Pruned {n_gaussians - self.get_xyz.shape[0]} points')

    def save_post_segmented_ply(self, path):
        seg_xyz = self._xyz.detach().cpu().numpy()

        bounds = self.bounds
        parts = self.parts
        print(f'Parts : {parts}')

        colors_lst = np.array([[255, 0, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255]])
        seg_colors = np.zeros((seg_xyz.shape[0], 3))
        group_idx = 0
        epsilon = np.ones(3) * 0.001
        for part in parts:
            for subpart in part:
                j = 0
                min, max = np.array(bounds[subpart]["min"]), np.array(bounds[subpart]["max"])
                # print(f'\nGroup {group_idx} - Part : {subpart} - Min : {min} - Max : {max}')
                for pid, point in enumerate(seg_xyz):
                    if np.all(point >= (min-epsilon)) and np.all(point <= (max+epsilon)):
                        seg_colors[pid] = colors_lst[group_idx]
                        j += 1
                # print(f"Subpart {subpart} has {j} points")
            group_idx += 1

        storePly(path, seg_xyz, seg_colors)
        
        # bound_color = np.array([255, 255, 255])

        # for key, bound in bounds.items():
        #     min = bound["min"]
        #     max = bound["max"]
        #     bound_xyz = np.array([[min[0], min[1], min[2]], [max[0], min[1], min[2]], [max[0], max[1], min[2]], [min[0], max[1], min[2]], [min[0], min[1], max[2]], [max[0], min[1], max[2]], [max[0], max[1], max[2]], [min[0], max[1], max[2]]])
        #     bound_colors = np.full((bound_xyz.shape[0], 3), bound_color)
        #     bound_path = path.replace(".ply", f"_{key}.ply")
        #     storePly(bound_path, bound_xyz, bound_colors)
    
    def save_segmented_ply(self, path):
        xyz = self._xyz.detach().cpu().numpy()
        print(f"The minimum group index is {torch.min(self._groups)}")

        groups = self._groups.detach().cpu().numpy()

        colors_lst = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]])
        colors = np.zeros((xyz.shape[0], 3))
        for group_idx in range(len(self.parts)):
            group_mask = groups == group_idx
            color = colors_lst[group_idx]
            colors[group_mask] = color

        storePly(path, xyz, colors)

#################################################################################################
#################################################################################################
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale

        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")



    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1