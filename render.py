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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
import colorsys
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, MLPModel

def render_set(model_path, name, iteration, views, gaussians, mlp, pipeline, background, recolor):
    if recolor[0] != -1:
        method_name = ("HSV_{}_" if recolor[4] == 0 else "RGB_{}_").format(int(recolor[0])) + "_".join(['{:.2f}'.format(x) for x in recolor[1:4]])
    else:
        method_name = "original"
    render_path = os.path.join(model_path, name, method_name, "renders")
    gts_path = os.path.join(model_path, name, method_name, "gt")
    layers_path = os.path.join(model_path, name, method_name, "layers")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(layers_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        result = render(view, gaussians, mlp, pipeline, background,
                        decompose_layer=(pipeline.color_compute_mode == "palette"), use_specular=True, recolor=recolor)
        rendering = result["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        #BHY 保存分层渲染结果
        if pipeline.color_compute_mode == "palette":
            layers = result["layers"]
            for name, layer in layers.items():
                torchvision.utils.save_image(layer, os.path.join(layers_path, '{0:05d}_'.format(idx) + name +".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, recolor):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        mlp = MLPModel(dataset.mlp_degree)
        scene = Scene(dataset, gaussians, mlp, load_iteration=iteration, shuffle=False)
        if recolor[0] != -1:
            if recolor[4] == 0:
                target_idx = int(recolor[0])
                hsv_target = colorsys.rgb_to_hsv(*recolor[1:4])
                hsv_original = colorsys.rgb_to_hsv(*gaussians.get_palette[target_idx].tolist())
                recolor[1], recolor[2], recolor[3] = hsv_target[0] - hsv_original[0], hsv_target[1] / hsv_original[1], hsv_target[2] / hsv_original[2]
                print("Edit palette[{}] with diff {} in HSV space".format(target_idx, recolor[1:4]))
            elif recolor[4] == 1:
                gaussians._palette[int(recolor[0])] = torch.tensor(recolor[1:4]).cuda()
                print("Replace palette[{}] with {} in RGB space".format(int(recolor[0]), recolor[1:4]))
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, mlp, pipeline, background, recolor)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, mlp, pipeline, background, recolor)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    #BHY recolor 参数：palette_idx(-1:don't recolor), new_r, new_g, new_b, mode(0:repalce, 1:modify in hsv)
    parser.add_argument("--recolor", nargs=5, type=float, default=[-1, 0.0, 0.0, 0.0, 0])
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.recolor)