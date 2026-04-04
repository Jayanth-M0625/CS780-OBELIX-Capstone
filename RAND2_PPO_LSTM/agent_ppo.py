import numpy as np
import torch
import torch.nn as nn
import os

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

class PPOAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(18, 64),
            nn.Tanh(),
        )
        self.policy = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 5)
        )

    def forward(self, x):
        x = self.shared(x)
        return self.policy(x)


_model = None

def load_model():
    global _model
    if _model is None:
        _model = PPOAgent()
        _model.load_state_dict(torch.load("ppo_weights.pth", map_location="cpu"))
        _model.eval()


@torch.no_grad()
def policy(obs, rng):
    load_model()
    x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    logits = _model(x)
    probs = torch.softmax(logits, dim=-1).numpy()[0]

    action = int(rng.choice(len(ACTIONS), p=probs))
    return ACTIONS[action]