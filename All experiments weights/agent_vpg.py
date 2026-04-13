import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

class VPG(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(18, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.policy = nn.Linear(64, 5)
        self.value = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc(x)
        return self.policy(x)


_model = None

def _load_once():
    global _model
    if _model is not None:
        return

    path = os.path.join(os.path.dirname(__file__), "weights.pth")

    _model = VPG()
    _model.load_state_dict(torch.load(path, map_location="cpu"))
    _model.eval()


@torch.no_grad()
def policy(obs, rng):
    _load_once()

    x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    logits = _model(x)

    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    action = int(np.argmax(probs))

    return ACTIONS[action]