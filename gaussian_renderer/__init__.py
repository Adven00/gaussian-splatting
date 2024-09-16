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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.mlp_model import MLPModel
from utils.sh_utils import eval_sh
from utils.palette_utils import *

#BHY 用 decompose_layer 控制是否分层渲染各种 gaussian 参数
def render(viewpoint_camera, pc : GaussianModel, mlp : MLPModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None,
           decompose_layer=False, use_specular=False, recolor_target=[-1, 0, 0, 0]):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    specular_precomp = None
    #BHY 不同 layer 的 colors_precomp, 收集给各 layer 渲染用
    colors_precomp_dict = {}
    if override_color is None:
        if pipe.color_compute_mode == "sh_python":
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        elif pipe.color_compute_mode == "sh_cuda":
            shs = pc.get_features
        #BHY palette 计算在这里
        elif pipe.color_compute_mode == "palette":
            palette_weights = palette_weights_from_alpha(pc.get_alpha)
            palette = pc.get_palette

            if use_specular:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)

                palette_offset = torch.bmm(
                    shs_view,
                    mlp(dir_pp_normalized).view(-1, 16, palette.shape[0]).to(torch.float32),
                ).squeeze().transpose(1, 2)

                soft_palette = palette_offset + palette

                if recolor_target[0] != -1:
                    idx = int(recolor_target[0])
                    hsv = rgb_to_hsv(soft_palette[:, idx])
                    hsv[:, 0] = (hsv[:, 0] + recolor_target[1]) % 1
                    hsv[:, 1:] = hsv[:, 1:] * torch.tensor(recolor_target[2:]).cuda()
                    rgb = hsv_to_rgb(hsv)
                    soft_palette[:, idx] = rgb

                colors_precomp = (palette_weights[:, None] @ soft_palette).squeeze()
                specular_precomp = (palette_weights[:, None] @ palette_offset).squeeze()
                colors_precomp_dict["specular"] = specular_precomp
            else:
                colors_precomp = palette_weights @ palette

            colors_precomp = torch.clamp(colors_precomp, 0.0, 1.0)

            #BHY 分解不同 layer 的 colors_precomp
            if decompose_layer:
                with torch.no_grad(): 
                    for i in range(palette.shape[0]):
                        new_palette_weights = torch.zeros_like(palette_weights, device="cuda")
                        new_palette_weights[:, i] = palette_weights[:, i]

                        if use_specular:
                            colors_precomp_dict["layer{}".format(i)] = (new_palette_weights[:, None] @ soft_palette).squeeze()
                        else:
                            colors_precomp_dict["layer{}".format(i)] = new_palette_weights @ palette
        else:
            assert False, "Invalid color compute mode {}!".format(pipe.color_compute_mode)
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    result = {
        "render": rendered_image,
        "specular_precomp": specular_precomp,
        "viewspace_points": screenspace_points,
        "visibility_filter" : radii > 0,
        "radii": radii
    }
    
    #BHY 渲染分解后的各 layer，开启 decompose_layer 才能启用
    if pipe.color_compute_mode == "palette" and decompose_layer:
        with torch.no_grad():    
            layers = {}
            for name, colors_precomp in colors_precomp_dict.items():
                layer, _ = rasterizer(
                    means3D = means3D,
                    means2D = means2D,
                    shs = shs,
                    colors_precomp = colors_precomp,
                    opacities = opacity,
                    scales = scales,
                    rotations = rotations,
                    cov3D_precomp = cov3D_precomp
                )
                layers[name] = layer
            result["layers"] = layers

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return result
