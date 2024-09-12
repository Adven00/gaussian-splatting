import json
import tinycudann as tcnn
import torch
import numpy as np
import os

class MLPModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        with open("scene/mlp_config.json") as config_file:
            self.config = json.load(config_file)
        # self.palette_size = -1
    
    def forward(self, dir_pp_normalized):
        out_basis = self.model(dir_pp_normalized)
        return out_basis

    def initialize(self):
        # if os.path.exists(palette_path):
        #     palette = torch.from_numpy(np.load(palette_path)).cuda()
        #     self.palette_size = palette.shape[0]
        # if self.palette_size == -1:
        #     return

        print("MLP output dim : {}".format(16))

        self.model = tcnn.NetworkWithInputEncoding(
            n_input_dims=3, n_output_dims=16,
            encoding_config=self.config["encoding"], 
            network_config=self.config["network"]
        ).to(torch.device("cuda"))

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.zeros_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def capture(self):
        return {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
