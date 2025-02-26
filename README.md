# Palette Gaussian

## 训练

所有参数使用默认值

`python train.py -s <path_to_dataset> --iterations 10000 --eval`

dataset 目录下要有 `rgb_palette.npy` 文件，可以用 `utils/palette_utils.py` 中的 `plot_palette_colors` 函数查看，训练完成后输出目录下也会有优化后的 palette 图片。

由于权重自由度的原因，palette 最后一个颜色一定是黑色且不参与优化。如果自定义 palette 要注意这一点。

高斯球颜色组成大致如下
$$
C= \sum_i \omega_i (P_i+\sum_j f_j \cdot \text{NB}_j(dir))
$$

相比 3dgs 增加的参数有

```c
// palette 参数相关的学习率
--alpha_lr  0.0005
--palette_lr  0.001

// neural baisi 的个数
--mlp_degree 8

// smooth L_0 loss 的 sigma 下降速度
// 参见论文 Fast Sparse Representation Based on Smoothed ℓ0 Norm
// 这个参数对分离效果影响很大
--sparsity_interval 1000

// 相关 loss 的系数
// 如果某个 loss 不想使用，设 lambda 为负数
--lambda_palette_loss  0.01
--lambda_specular_loss  -1
--lambda_sparsity_loss  0.02

// 从哪一次迭代开始这些操作
// 如果不想使用，设成一个很大的数即可
--palette_from_iter  1000 // 将 palette 本身加入优化
--specular_from_iter  7000 // 为每个高斯球增加各向异性高光（占用大量显存）
```

## 渲染和重着色

`python render.py -m .\output\c340b1fd-6 --eval --recolor 2 0 0 1`

相比 3dgs 增加的参数有

```c
// 训练完成后输出目录下会有优化后的 palette 图片
// 将 index 位置（从 0 起）的 palette color 重着色为 r g b
// mode 是重着色模式，0 为 hsv 空间，1 为 rgb 直接替换，2 为 lab 空间
// object_id 是需要重着色的物体的编码，默认 -1 为全重着色
// 参见 PaletteNeRF Suplementary
--recolor <index r g b mode object_id>
```

## palette 提取

`python extract.py -m .\output\garden\3D-GS --normalize`

注意，只能从训练好的，未使用 palette 的高斯点云中提取（`color_compute_mode = sh_cuda`），颜色来自 0 阶球谐。输出 `extracted_rgb_palette.jpg` 和 `extracted_rgb_palette.npy` 。

## object mask 相关
使用 gaussian grouping 中经过修改的 `diff-gaussian-rasterization` 模块，可以将高斯球中的特征向量（16 维）渲染成特征图，与颜色的渲染方式一样

```python
rendered_image, radii, rendered_objects = rasterizer(
    means3D = means3D,
    means2D = means2D,
    shs = shs,
    sh_objs = sh_objs,
    colors_precomp = colors_precomp,
    opacities = opacity,
    scales = scales,
    rotations = rotations,
    cov3D_precomp = cov3D_precomp)
```

`classifier` 将 16 维特征向量图转化成 256 维类别向量图，再取最大值索引，再去重

```python
classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
logits = classifier(rendering_obj)
obj = torch.argmax(logits, dim=0)
all_obj_ids = np.unique(obj)
```
