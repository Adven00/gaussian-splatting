import torch
import tinycudann as tcnn
import commentjson as json
import torch.nn as nn
import torch.nn.functional as F

class PaletteModel(nn.Module):
    def __init__(self):

        super(PaletteModel, self).__init__()

        with open("palette/config.json") as f:
            config = json.load(f)

        with open("palette/palette.txt") as f:
            palette = [torch.tensor([float(a) for a in line.split(' ')]) for line in f.readlines()]
            palette = torch.vstack(palette)

        self.palette = nn.Parameter(palette).cuda()
        self.palette_feat_dim = 16
        self.palette_dim = palette.shape[0]
        # just follow PaletteNeRF
        # self.weight_bias = 0.05

        self.pos_encoding = tcnn.Encoding(3, config["pos_encoding"])
        self.dir_encoding = tcnn.Encoding(3, config["dir_encoding"])

        self.specular_mlp = tcnn.Network(self.pos_encoding.n_output_dims + self.dir_encoding.n_output_dims, 3, config["specular_mlp"])
        self.diffuse_mlp = tcnn.Network(self.pos_encoding.n_output_dims, 3, config["diffuse_mlp"])
        self.palette_mlp = tcnn.Network(self.pos_encoding.n_output_dims + 3, self.palette_feat_dim, config["palette_mlp"])
        self.weight_mlp = tcnn.Network(self.palette_feat_dim, self.palette_dim, config["weight_mlp"])
        self.offest_intensity_mlp = tcnn.Network(self.palette_feat_dim, self.palette_dim * 3 + 1, config["offest_intensity_mlp"])

    def forward(self, pos, dir):
        pos_feat = self.pos_encoding(pos)
        dir_feat = self.dir_encoding(dir)

        specular = self.specular_mlp(torch.cat([pos_feat, dir_feat], dim=-1))
        diffuse = self.diffuse_mlp(pos_feat)

        palette_feat = self.palette_mlp(torch.cat([diffuse.detach(), pos_feat], dim=-1))

        weight = self.weight_mlp(palette_feat)
        weight = weight / (weight.sum(dim=-1, keepdim=True))

        offset_intensity = self.offest_intensity_mlp(palette_feat)
        offset, intensity = offset_intensity[..., :-1], offset_intensity[..., -1:]
        intensity = F.softplus(intensity)
        offset = offset.reshape((-1, self.palette_dim, 3))

        return intensity, offset, weight, diffuse, specular
    
    def compose(self, intensity, offset, weight, specular):
        return ((weight.repeat(1, 3).reshape(-1, 3, self.palette_dim).transpose(1, 2) * (self.palette + offset)).sum(dim=1)) \
                * intensity.repeat(1, 3) + specular
    
if __name__ == "__main__":
    pos = torch.rand((6, 3)).cuda()
    dir = torch.rand((6, 3)).cuda()

    pm = PaletteModel()
    intensity, offset, weight, diffuse, specular = pm(pos, dir)
    color = pm.compose(intensity, offset, weight, specular)

    print(color)