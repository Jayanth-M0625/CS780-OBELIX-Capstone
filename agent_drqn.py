"""DRQN agent scaffold for OBELIX (CPU).

This agent uses a Recurrent Dueling Network to handle POMDP environments.
It loads weights from weights_drqn.pth.

Submission ZIP structure:
  submission.zip
    agent_drqn.py (rename to agent.py for submission)
    weights_drqn.pth (rename to weights.pth for submission)
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]

class DRQN(nn.Module):
    def __init__(self, in_dim=18, n_actions=5, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Feature extractor
        self.feature = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Dueling heads
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x, hidden=None):
        # x shape: (batch, seq_len, in_dim)
        b, s, d = x.shape
        f = self.feature(x.view(-1, d)) # (b*s, hidden_dim)
        f = f.view(b, s, -1)
        
        lstm_out, hidden = self.lstm(f, hidden)
        
        # Take the last output for Q-values if we are doing single-step policy
        # Or if we want all Q-values in sequence for training
        last_out = lstm_out # (b, s, hidden_dim)
        
        v = self.value(last_out) # (b, s, 1)
        a = self.advantage(last_out) # (b, s, n_actions)
        
        q = v + a - a.mean(dim=2, keepdim=True)
        return q, hidden

    def init_hidden(self, batch_size=1):
        return (torch.zeros(1, batch_size, self.hidden_dim),
                torch.zeros(1, batch_size, self.hidden_dim))

_model: Optional[DRQN] = None
_hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
_LAST_RNG_ID: Optional[int] = None
_steps_since_signal: int = 0

def _load_once():
    global _model
    if _model is not None:
        return
    here = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights_drqn.pth")
    if not os.path.exists(wpath):
        wpath = os.path.join(here, "weights.pth")
    
    if not os.path.exists(wpath):
        wpath = "weights_drqn.pth"

    m = DRQN()
    if os.path.exists(wpath):
        sd = torch.load(wpath, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        m.load_state_dict(sd, strict=True)
    
    m.eval()
    _model = m

@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _hidden_state, _LAST_RNG_ID, _model, _steps_since_signal
    _load_once()
    
    # Detect new episode via rng id
    current_rng_id = id(rng)
    if _LAST_RNG_ID != current_rng_id:
        _hidden_state = _model.init_hidden(1)
        _LAST_RNG_ID = current_rng_id
        _steps_since_signal = 0
    
    # Check signal
    # near(0,2, 4,6,8,10, 12,14) + IR(16)
    # The obs has 18 elements.
    any_signal = np.any(obs > 0)
    
    if not any_signal:
        _steps_since_signal += 1
        # Gradual Search Bias: FW probability ramps up by 4% each step
        fw_prob = min(0.8, 0.2 + 0.04 * _steps_since_signal)
        other_prob = (1.0 - fw_prob) / 4.0
        probs = [other_prob, other_prob, fw_prob, other_prob, other_prob]
        
        # Select action using bias
        best = int(rng.choice(len(ACTIONS), p=probs))
        
        # Feed the "empty" observation into LSTM anyway to keep zero-history correctly
        x = torch.tensor(obs, dtype=torch.float32).view(1, 1, -1)
        _, _hidden_state = _model(x, _hidden_state)
    else:
        _steps_since_signal = 0
        # Normal DRQN logic
        x = torch.tensor(obs, dtype=torch.float32).view(1, 1, -1)
        q_out, _hidden_state = _model(x, _hidden_state)
        q = q_out.squeeze().cpu().numpy()
        best = int(np.argmax(q))
    
    return ACTIONS[best]
