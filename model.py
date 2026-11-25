import torch
import torch.nn as nn


class CFTRMarkerNet(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.layers(x)


def clone_state_dict(state_dict, target_device="cpu"):
    return {k: v.detach().clone().to(target_device) for k, v in state_dict.items()}
