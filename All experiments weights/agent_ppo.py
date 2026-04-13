from __future__ import annotations
from typing import List, Optional
import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]

# =========================
# PPO Network (same as training)
# =========================
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

        self.value = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.shared(x)
        return self.policy(x)  # only policy used


# =========================
# Globals (same pattern as template)
# =========================
_model: Optional[PPOAgent] = None
_last_action: Optional[int] = None
_repeat_count: int = 0

_MAX_REPEAT = 2
_CLOSE_PROB_DELTA = 0.05


# =========================
# Load once (submission style)
# =========================
def _load_once():
    global _model
    if _model is not None:
        return

    here = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")

    if not os.path.exists(wpath):
        raise FileNotFoundError("weights.pth not found next to agent.py")

    m = PPOAgent()
    sd = torch.load(wpath, map_location="cpu")

    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    m.load_state_dict(sd, strict=True)
    m.eval()
    _model = m


# =========================
# Policy (deterministic + smoothing)
# =========================
@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _last_action, _repeat_count
    _load_once()

    x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    logits = _model(x)

    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    # GREEDY (important for evaluation stability)
    best = int(np.argmax(probs))

    # ===== smoothing (avoid oscillation) =====
    if _last_action is not None:
        order = np.argsort(-probs)
        best_p, second_p = float(probs[order[0]]), float(probs[order[1]])

        if (best_p - second_p) < _CLOSE_PROB_DELTA:
            if _repeat_count < _MAX_REPEAT:
                best = _last_action
                _repeat_count += 1
            else:
                _repeat_count = 0
        else:
            _repeat_count = 0

    _last_action = best
    return ACTIONS[best]