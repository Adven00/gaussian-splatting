import json
import tinycudann as tcnn
import torch
import numpy as np
import os

class MLPModel(torch.nn.Module):
    def __init__(self, mlp_degree):
        super().__init__()
        with open("scene/mlp_config.json") as config_file:
            self.config = json.load(config_file)
        self.palette_size = -1
        self.mlp_degree = mlp_degree
    
    def forward(self, dir_pp_normalized):
        out_basis = self.model(dir_pp_normalized)
        return out_basis

    def initialize(self, palette_path):
        if os.path.exists(palette_path):
            palette = torch.from_numpy(np.load(palette_path)).cuda()
            self.palette_size = palette.shape[0]
        else:
            return

        print("MLP output dim : {}".format((self.palette_size - 1) * self.mlp_degree))

        self.model = tcnn.NetworkWithInputEncoding(
            n_input_dims=3, n_output_dims=(self.palette_size - 1) * self.mlp_degree,
            encoding_config=self.config["encoding"], 
            network_config=self.config["network"]
        ).to(torch.device("cuda"))

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
