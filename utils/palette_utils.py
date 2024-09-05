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
def colors_from_palette_model(palette, weight, offset):
    return (weight[:, None] @ (palette + offset)).squeeze()