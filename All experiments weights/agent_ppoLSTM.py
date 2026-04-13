from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

# =========================
# SAME AS TRAINING
# =========================
class PPO_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(18, 64)
        self.lstm = nn.LSTM(64, 64, batch_first=True)
        self.policy = nn.Linear(64, 5)
        self.value = nn.Linear(64, 1)   # ✅ REQUIRED

    def forward(self, x, hx, cx):
        x = torch.tanh(self.fc(x))
        x, (hx, cx) = self.lstm(x, (hx, cx))
        logits = self.policy(x)
        value = self.value(x)
        return logits, value, hx, cx


# =========================
# GLOBALS
# =========================
_model = None
_hx = None
_cx = None


# =========================
# LOAD FUNCTION (for metrics.py)
# =========================
def load(weights_path):
    global _model, _hx, _cx

    _model = PPO_LSTM()
    _model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    _model.eval()

    _hx = torch.zeros(1, 1, 64)
    _cx = torch.zeros(1, 1, 64)


# =========================
# FALLBACK (for submission)
# =========================
def _load_once():
    global _model, _hx, _cx

    if _model is not None:
        return

    here = os.path.dirname(__file__)
    path = os.path.join(here, "weights.pth")

    _model = PPO_LSTM()
    _model.load_state_dict(torch.load(path, map_location="cpu"))
    _model.eval()

    _hx = torch.zeros(1, 1, 64)
    _cx = torch.zeros(1, 1, 64)


# =========================
# POLICY
# =========================
@torch.no_grad()
def policy(obs, rng):
    global _hx, _cx

    _load_once()  # works for submission OR metrics

    x = torch.tensor(obs, dtype=torch.float32).view(1, 1, -1)
    logits, _, _hx, _cx = _model(x, _hx, _cx)

    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0, 0]
    action = int(np.argmax(probs))

    return ACTIONS[action]