import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt

#BHY 参考 recolornerf
def palette_weights_from_alpha(alpha):
    log_a = torch.log(alpha)
    log_wa = torch.cumsum(torch.log(1 - alpha), dim=-1)
    w_0 = alpha[..., :1]
    w_a = torch.exp(log_wa[..., :-1] + log_a[..., 1:])
    w_last = torch.exp(log_wa[..., -1:])
    palette_weights = torch.cat((w_0, w_a, w_last), dim=-1)
    # palette_weights guarantee sum to 1
    # assert torch.allclose(palette_weights.sum(dim=-1), torch.ones(()), atol=1e-3)
    return palette_weights


def plot_palette_colors(palette, c=50):
    palette2 = np.ones((1 * c, len(palette) * c, 3))
    for i in range(len(palette)):
        palette2[:, i * c:i * c + c, :] = palette[i, :].reshape((1, 1, -1))
    
    plt.figure()
    plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    plt.imshow(palette2)

# import torch
# 
# palette_offset = torch.zeros([1000, 6, 3])
# palette = torch.zeros([6, 3])
# weight = torch.zeros([1000, 6])
# 
# new_palette = palette_offset + palette
# 
# print(weight[:, None].shape)
# print(new_palette.shape)
# 
# color = weight[:, None] @ new_palette
# 
# print(color.shape)
# color = color.squeeze()
# print(color.shape)
def colors_from_palette(palette, weight, offset, use_offset):
    palette = torch.clamp(palette, 0.0, 1.0)
    if use_offset:
        colors = (weight[:, None] @ (palette + offset)).squeeze()
    else:
        colors = weight @ palette
    colors = torch.clamp(colors, 0.0, 1.0) 
    return colors

def rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsv_h = torch.empty_like(rgb[:, 0:1])
    cmax_idx[delta == 0] = 3
    hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsv_h[cmax_idx == 3] = 0.
    hsv_h /= 6.
    hsv_s = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)
    hsv_v = cmax
    return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)


def hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    hsv_h, hsv_s, hsv_l = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    _c = hsv_l * hsv_s
    _x = _c * (- torch.abs(hsv_h * 6. % 2. - 1) + 1.)
    _m = hsv_l - _c
    _o = torch.zeros_like(_c)
    idx = ((hsv_h * 6.).type(torch.uint8) % 6).squeeze()
    rgb = torch.empty_like(hsv)
    rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
    rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
    rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
    rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
    rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
    rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
    rgb += _m
    return rgb