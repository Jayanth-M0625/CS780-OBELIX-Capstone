"""Better DQN-style agent scaffold for OBELIX (CPU).

This agent is *evaluation-only*: it loads pretrained weights from a file
placed next to agent.py inside the submission zip (weights.pth).

Why your STD is huge:
- if the policy is stochastic (epsilon > 0) during evaluation, scores vary a lot.
Fix:
- greedy action selection (epsilon=0), model.eval(), torch.no_grad().
- optional action smoothing to reduce oscillation when Q-values are close.

Submission ZIP structure:
  submission.zip
    agent.py
    weights.pth
"""

from __future__ import annotations
from typing import List, Optional
import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]

class D3QN(nn.Module):
    def __init__(self, in_dim=18, n_actions=5):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        f = self.feature(x)
        v = self.value(f)
        a = self.advantage(f)
        return v + a - a.mean(dim=1, keepdim=True)
_model: Optional[D3QN] = None
_last_action: Optional[int] = None
_repeat_count: int = 0

_MAX_REPEAT = 2
_CLOSE_Q_DELTA = 0.05

def _sensors_active(s: np.ndarray) -> bool:
    near_right = s[[0, 2]]
    near_front = s[[4, 6, 8, 10]]
    near_left  = s[[12, 14]]
    ir = s[16]
    return (np.sum(near_right) +
            np.sum(near_front) +
            np.sum(near_left) +
            ir) > 0

def biased_random_action(rng):
    # Bias toward forward movement
    probs = np.array([0.1, 0.1, 0.6, 0.1, 0.1])  # L45, L22, FW, R22, R45
    return ACTIONS[int(rng.choice(len(ACTIONS), p=probs))]

def _load_once():
    global _model
    if _model is not None:
        return
    here = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")
    if not os.path.exists(wpath):
        raise FileNotFoundError(
            "weights.pth not found next to agent.py. Train offline and include it in the submission zip."
        )
    m = D3QN()
    sd = torch.load(wpath, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    m.load_state_dict(sd, strict=True)
    m.eval()
    _model = m

@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _last_action, _repeat_count
    _load_once()

    if not _sensors_active(obs):
        return biased_random_action(rng)

    x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    q = _model(x).squeeze(0).cpu().numpy()
    best = int(np.argmax(q))

    if _last_action is not None:
        order = np.argsort(-q)
        best_q, second_q = float(q[order[0]]), float(q[order[1]])
        if (best_q - second_q) < _CLOSE_Q_DELTA:
            if _repeat_count < _MAX_REPEAT:
                best = _last_action
                _repeat_count += 1
            else:
                _repeat_count = 0
        else:
            _repeat_count = 0

    _last_action = best
    return ACTIONS[best]



