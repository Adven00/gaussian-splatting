import torch
from scene import Scene
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, MLPModel
from utils.sh_utils import eval_sh
import os
import numpy as np
import random
import _palette_func
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

from rgbsg import *

def run_kmeans(
    n_clusters: int,
    points: np.ndarray,
    init: np.ndarray,
    sample_weight: np.ndarray
):
    kmeans = KMeans(n_clusters=n_clusters, init=init).fit(X=points, sample_weight=sample_weight)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    center_weights = np.zeros(n_clusters)
    for i in range(n_clusters):
        center_weights[i] = np.sum(sample_weight[labels==i])

    idcs = np.argsort(center_weights * -1)

    return centers[idcs], center_weights[idcs]

def compute_RGB_histogram(
    colors_rgb: np.ndarray,
    weights: np.ndarray,
    bits_per_channel: int
):
    assert colors_rgb.ndim == 2 and colors_rgb.shape[1] == 3
    assert weights.ndim == 1
    assert len(colors_rgb) == len(weights)
    assert 1 <= bits_per_channel and bits_per_channel <=8

    try:
        bin_weights, bin_centers_rgb = _palette_func.compute_RGB_histogram(
            colors_rgb.flatten(), weights.flatten(), bits_per_channel)
    except RuntimeError as err:
        assert False

    return bin_weights, bin_centers_rgb

def palette_extraction(
    colors,
    tau: float = 8e-3,
    palette_size = None,
    error_thres = 5.0 / 255.0
):
    weights = np.ones_like(colors[...,0])
    colors = colors.reshape(-1,3)
    weights = weights.flatten()

    assert len(weights[weights < 0]) == 0, 'negative weight indicates the failure of radiance sampling'

    ## save radiance samples (outside timing analysis)
    res = 800
    n_total = res**2
    random.seed(0)
    idcs = random.sample(range(len(colors)), n_total)
    assert len(idcs) == len(set(idcs)), 'each element of idcs should be unique'

    ## coarse histogram (2^3 = 8 bins)
    bin_weights_coarse, bin_centers_coarse = compute_RGB_histogram(colors, weights, bits_per_channel=3)
    sum_weights = np.sum(bin_weights_coarse)
    bin_weights_coarse /= sum_weights

    idcs = bin_weights_coarse > tau
    bin_weights_coarse = bin_weights_coarse[idcs]
    bin_centers_coarse = bin_centers_coarse[idcs]

    ## fine histogram (2^5 = 32 bins)
    bin_weights_fine, bin_centers_fine = compute_RGB_histogram(colors, weights, bits_per_channel=5)
    idcs = bin_weights_fine > 0
    bin_weights_fine = bin_weights_fine[idcs]
    bin_weights_fine /= sum_weights
    bin_centers_fine = bin_centers_fine[idcs]

    centers, center_weights = run_kmeans(
        n_clusters=len(bin_weights_coarse), points=bin_centers_fine,
        init=bin_centers_coarse, sample_weight=bin_weights_fine)

    ## convex hull simplification
    rgb_palette = Hull_Simplification_posternerf(
        centers.astype(np.double), "./tmp/test",
        pixel_counts=center_weights,
        error_thres=error_thres,
        target_size=palette_size)
    
    return rgb_palette

def extract(dataset : ModelParams, iteration : int, normalize : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        mlp = MLPModel(dataset.mlp_degree)
        scene = Scene(dataset, gaussians, mlp, load_iteration=iteration, shuffle=False)

        shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree+1)**2)
        #BHY 只算 0 阶球谐颜色，所以 deg=0，dirs 无用
        sh2rgb = eval_sh(deg=0, sh=shs_view, dirs=[])
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

        if normalize:
            colors_precomp_norm = colors_precomp + 0.1
            colors_precomp_norm = colors_precomp_norm / colors_precomp_norm.norm(dim=-1, p=2, keepdim=True)
            return palette_extraction(colors_precomp_norm.cpu().numpy())
        else:
            return palette_extraction(colors_precomp.cpu().numpy())


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    args = get_combined_args(parser)
    print("Extracting " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    palette_path = os.path.join(args.model_path, "extracted_rgb_palette.npy")
    palette_fig_path = os.path.join(args.model_path, "extracted_rgb_palette.jpg")
    palette = extract(model.extract(args), args.iteration, args.normalize)

    np.save(palette_path, palette)
    palette2 = np.ones((1 * 50, len(palette) * 50, 3))
    for i in range(len(palette)):
        palette2[:, i * 50:i * 50 + 50, :] = palette[i, :].reshape((1, 1, -1))

    plt.figure()
    plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    plt.imshow(palette2)
    plt.savefig(palette_fig_path)