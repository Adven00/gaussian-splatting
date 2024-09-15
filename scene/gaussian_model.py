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
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.palette_utils import rgb_to_hsv, hsv_to_rgb
from matplotlib import pyplot as plt

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
        self.alpha_activation = torch.sigmoid
        self.offset_activation = torch.tanh
        self.inverse_opacity_activation = inverse_sigmoid
        self.intensity_activation = torch.nn.functional.softplus

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int, palette_size : int = -1):
        self.active_sh_degree = 0
        #BHY 增加参数 palette_size，为 -1 说明不使用 palette
        self.palette_size = palette_size
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        #BHY 先放一个空的 alpha 和 palette 在这里
        # gaussian 中放 palette offset 试试？
        # palette 不在 gaussian model 中优化
        self._alpha = torch.empty(0)
        self._palette = torch.empty(0)
        self._palette_offset = torch.empty(0)
        self._intensity = torch.empty(0)

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

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._palette,
            self._alpha,
            self._palette_offset,
            self._intensity,
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
        self._alpha,
        self._palette,
        self._palette_offset,
        self._intensity,
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
    
    #BHY 用 sigmoid 保证输出 [0, 1]
    @property
    def get_alpha(self):
        return self.alpha_activation(self._alpha)
    
    @property
    def get_palette_offset(self):
        # return self._palette_offset
        return torch.cat((self.offset_activation(self._palette_offset) / 2, 
                          torch.zeros([self._palette_offset.shape[0], 1, 3], dtype=torch.float, device="cuda")), dim=1) 
    
    #BHY sigmoid(-x) = 1 - sigmoid(x)
    # @property
    # def get_one_minus_alpha(self):
    #     return self.alpha_activation(-self._alpha)
    
    @property
    def get_palette(self):
        return torch.cat((self._palette, torch.tensor([[0, 0, 0]], dtype=torch.float, device="cuda")), dim=0)
    
    @property
    def get_intensity(self):
        return self.intensity_activation(self._intensity)
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    #BHY 实际的初始化发生在这里
    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, palette_path):
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

        #BHY 加载 palette，如有，注意 palette 末尾的黑色不参与优化
        #BHY 转化到 hsv 色彩空间做优化，hue 不参与优化？
        if os.path.exists(palette_path):
            palette = torch.from_numpy(np.load(palette_path))[:-1].cuda()
            self._palette = nn.Parameter(palette.requires_grad_(True))
            
            self.palette_size = self._palette.shape[0] + 1
            print("Number of palette colors : {}".format(self.palette_size))

        #BHY 暂时先初始化为 0，把 alpha 转化成可优化的参数，注意 alpha 与 feature 可以共存
        if self.palette_size != -1:
            #BHY 注意这里的 alpha shape，默认最后一层是背景颜色，alpha 为 1！
            alpha = torch.zeros((fused_point_cloud.shape[0], self.palette_size - 1), device="cuda")
            self._alpha = nn.Parameter(alpha.requires_grad_(True))
            #BHY 初始化 palette_offset
            palette_offset = torch.zeros((fused_point_cloud.shape[0], self.palette_size - 1, 3), device="cuda")
            self._palette_offset = nn.Parameter(palette_offset.requires_grad_(True))
            #BHY 初始化 intensity
            intensity = torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
            self._intensity = nn.Parameter(intensity.requires_grad_(True))

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

        #BHY 设置学习率
        if self.palette_size != -1:
            l.append({'params': [self._alpha], 'lr': training_args.alpha_lr, "name": "alpha"})
            l.append({'params': [self._intensity], 'lr': training_args.intensity_lr, "name": "intensity"})
            l.append({'params': [self._palette_offset], 'lr': training_args.palette_offset_lr, "name": "palette_offset"})

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


    '''
    https://paulbourke.net/dataformats/ply/

    ply example
    format ascii 1.0           { ascii/binary, format version number }
    comment made by Greg Turk  { comments keyword specified, like all lines }
    element vertex 8           { define "vertex" element, 8 of them in file }
    property float x           { vertex contains float "x" coordinate }
    property float y           { y coordinate is also a vertex property }
    property float z           { z coordinate, too }
    element face 6             { there are 6 "face" elements in the file }
    property list uchar int vertex_index { "vertex_indices" is a list of ints }
    end_header                 { delimits the end of the header }
    0 0 0                      { start of vertex list }
    0 0 1
    0 1 1
    0 1 0
    1 0 0
    1 0 1
    1 1 1
    1 1 0
    4 0 1 2 3                  { start of face list }
    4 7 6 5 4
    4 0 4 5 1
    4 1 5 6 2
    4 2 6 7 3
    4 3 7 4 0
    '''
    #BHY 构建 ply 的顶点属性列表
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
        #BHY 加入 palette 相关属性
        if self.palette_size != -1:
            for i in range(self._alpha.shape[1]):
                l.append('alpha_{}'.format(i))

            for i in range(self._palette_offset.shape[1]*3):
                l.append('p_offset_{}'.format(i))

            l.append('intensity')
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

        #BHY palette 相关属性保存到 ply
        if self.palette_size != -1:
            intensity = self._intensity.detach().cpu().numpy()
            alpha = self._alpha.detach().cpu().numpy()
            palette_offset = self._palette_offset.detach().flatten(start_dim=1).contiguous().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)

        if self.palette_size != -1:
            attributes = np.concatenate((attributes, alpha, palette_offset, intensity), axis=1)

        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def optimize_palette(self, lr):
        self.optimizer.add_param_group({'params': [self._palette], 'lr': lr, "name": "palette"})

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

        #BHY 读取 ply 中的 palette 相关属性
        if self.palette_size != -1:
            alpha_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("alpha_")]
            alpha_names = sorted(alpha_names, key = lambda x: int(x.split('_')[-1]))
            assert len(alpha_names)==self.palette_size - 1

            alpha = np.zeros((xyz.shape[0], len(alpha_names)))
            for idx, attr_name in enumerate(alpha_names):
                alpha[:, idx] = np.asarray(plydata.elements[0][attr_name])

            palette_offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("p_offset_")]
            palette_offset_names = sorted(palette_offset_names, key = lambda x: int(x.split('_')[-1]))
            assert len(palette_offset_names)==(self.palette_size - 1) * 3

            palette_offset = np.zeros((xyz.shape[0], len(palette_offset_names)))
            for idx, attr_name in enumerate(palette_offset_names):
                palette_offset[:, idx] = np.asarray(plydata.elements[0][attr_name])
            palette_offset = palette_offset.reshape((-1, self.palette_size - 1, 3))

            intensity = np.asarray(plydata.elements[0]["intensity"])[..., np.newaxis]

            self._intensity = nn.Parameter(torch.tensor(intensity, dtype=torch.float, device="cuda").requires_grad_(True))
            self._palette_offset = nn.Parameter(torch.tensor(palette_offset, dtype=torch.float, device="cuda").requires_grad_(True))
            self._alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float, device="cuda").requires_grad_(True))

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def load_palette(self, palette_path):
        #BHY 加载 palette，如有
        if os.path.exists(palette_path):
            palette = torch.from_numpy(np.load(palette_path))[:-1].cuda()
            self._palette = nn.Parameter(palette.requires_grad_(True))
            self.palette_size = self._palette.shape[0] + 1
            print("Number of palette colors : {}".format(self.palette_size))

    def save_palette(self, model_path):
        palette_path = os.path.join(model_path, "rgb_palette.npy")
        palette_fig_path = os.path.join(model_path, "rgb_palette.jpg")
        palette = self.get_palette.detach().cpu().numpy()
        np.save(palette_path, palette)

        palette2 = np.ones((1 * 50, len(palette) * 50, 3))
        for i in range(len(palette)):
            palette2[:, i * 50:i * 50 + 50, :] = palette[i, :].reshape((1, 1, -1))
    
        plt.figure()
        plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        plt.imshow(palette2)
        plt.savefig(palette_fig_path)

    def repalce_palette(self, index, target):
        self._palette[index] = torch.tensor(target)
        print("Repalce palette[{}] with {}".format(index, target))


    #BHY 目前只用于 opacity ，不用管
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
            if group["name"] == "palette":
                continue
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
        #BHY 与高斯球修剪相关
        if self.palette_size != -1:
            self._alpha = optimizable_tensors["alpha"]
            self._intensity = optimizable_tensors["intensity"]
            self._palette_offset = optimizable_tensors["palette_offset"]
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
            if group["name"] == "palette":
                continue
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

    #BHY 与高斯球的加密有关
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_alpha, new_intensity, new_palette_offset):
        d = {"xyz": new_xyz,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        if self.palette_size != -1:
            d["alpha"] = new_alpha
            d["intensity"] = new_intensity
            d["palette_offset"] = new_palette_offset
        d["f_dc"] = new_features_dc
        d["f_rest"] = new_features_rest

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]

        if self.palette_size != -1:
            self._alpha = optimizable_tensors["alpha"]
            self._intensity = optimizable_tensors["intensity"]
            self._palette_offset = optimizable_tensors["palette_offset"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]

        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    #BHY 分裂高斯球的函数，N 是分裂的个数，在第一个维度上复制 N 次即可
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

        if self.palette_size != -1:
            new_alpha = self._alpha[selected_pts_mask].repeat(N,1)
            new_intensity = self._intensity[selected_pts_mask].repeat(N,1)
            new_palette_offset = self._palette_offset[selected_pts_mask].repeat(N,1,1)
        else:
            new_alpha = None
            new_intensity = None
            new_palette_offset = None

        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_alpha, new_intensity, new_palette_offset)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_opacity = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        
        if self.palette_size != -1:
            new_alpha = self._alpha[selected_pts_mask]
            new_intensity = self._intensity[selected_pts_mask]
            new_palette_offset = self._palette_offset[selected_pts_mask]
        else:
            new_alpha = None
            new_intensity = None
            new_palette_offset = None

        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
            
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_alpha, new_intensity, new_palette_offset)

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