from __future__ import annotations
from typing import Optional
import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

class PPO_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(18, 64)
        self.lstm = nn.LSTM(64, 64, batch_first=True)
        self.policy = nn.Linear(64, 5)

    def forward(self, x, hx, cx):
        x = torch.tanh(self.fc(x))
        x, (hx, cx) = self.lstm(x, (hx, cx))
        logits = self.policy(x)
        return logits, hx, cx


_model: Optional[PPO_LSTM] = None
_hx, _cx = None, None


def _load_once():
    global _model, _hx, _cx
    if _model is not None:
        return

    here = os.path.dirname(__file__)
    path = os.path.join(here, "weights.pth")

    _model = PPO_LSTM()
    _model.load_state_dict(torch.load(path, map_location="cpu"))
    _model.eval()

    _hx = torch.zeros(1,1,64)
    _cx = torch.zeros(1,1,64)


@torch.no_grad()
def policy(obs, rng):
    global _hx, _cx
    _load_once()

    x = torch.tensor(obs, dtype=torch.float32).view(1,1,-1)
    logits, _hx, _cx = _model(x, _hx, _cx)

    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0,0]
    action = int(np.argmax(probs))  # deterministic

    return ACTIONS[action]