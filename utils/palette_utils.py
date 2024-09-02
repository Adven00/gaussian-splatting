import torch
import torch.nn.functional as F

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
    

# if __name__ == "__main__":
#     print(palette_weights_from_alpha(torch.sigmoid(torch.tensor([[1, 3, 5, 7, 9]]))))