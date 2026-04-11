"""Offline trainer: DRQN + Sequence Replay (CPU) for OBELIX.

Run locally to create weights_drqn.pth.
Example:
  python train_drqn.py --obelix_py ./obelix.py --out weights_drqn.pth --episodes 2000 --difficulty 2 --wall_obstacles
"""

from __future__ import annotations
print("Importing libraries...")
import argparse, random, os
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from reward_wrapper import RewardWrapper

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
print("Initializing classes...")
class DRQN(nn.Module):
    def __init__(self, in_dim=18, n_actions=5, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feature = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
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
        # x: (batch, seq_len, in_dim)
        b, s, d = x.shape
        f = self.feature(x.reshape(-1, d))
        f = f.view(b, s, -1)
        lstm_out, hidden = self.lstm(f, hidden)
        
        v = self.value(lstm_out)
        a = self.advantage(lstm_out)
        q = v + a - a.mean(dim=2, keepdim=True)
        return q, hidden

    def init_hidden(self, batch_size=1):
        return (torch.zeros(1, batch_size, self.hidden_dim),
                torch.zeros(1, batch_size, self.hidden_dim))

@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool

class SequenceReplay:
    def __init__(self, cap: int = 100_000):
        self.buf: Deque[Deque[Transition]] = deque(maxlen=cap)
        self.current_episode: Deque[Transition] = deque()

    def add_step(self, t: Transition):
        self.current_episode.append(t)
        if t.done:
            self.buf.append(list(self.current_episode))
            self.current_episode = deque()

    def sample(self, batch_size: int, seq_len: int):
        # Sample episodes that are long enough
        valid_episodes = [ep for ep in self.buf if len(ep) >= seq_len]
        if not valid_episodes:
            # Fallback to whatever we have if none are long enough yet
            valid_episodes = list(self.buf)
        
        batch_idx = np.random.choice(len(valid_episodes), size=batch_size, replace=True)
        
        s_batch, a_batch, r_batch, s2_batch, d_batch = [], [], [], [], []
        
        for i in batch_idx:
            ep = valid_episodes[i]
            if len(ep) > seq_len:
                start = np.random.randint(0, len(ep) - seq_len + 1)
            else:
                start = 0
            
            chunk = ep[start:start+seq_len]
            
            s_batch.append([t.s for t in chunk])
            a_batch.append([t.a for t in chunk])
            r_batch.append([t.r for t in chunk])
            s2_batch.append([t.s2 for t in chunk])
            d_batch.append([t.done for t in chunk])
            
        return (torch.tensor(np.array(s_batch), dtype=torch.float32),
                torch.tensor(np.array(a_batch), dtype=torch.long),
                torch.tensor(np.array(r_batch), dtype=torch.float32),
                torch.tensor(np.array(s2_batch), dtype=torch.float32),
                torch.tensor(np.array(d_batch), dtype=torch.float32))

    def __len__(self): return len(self.buf)

def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX
print("Starting main training...")
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out", type=str, default="weights_drqn.pth")
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--max_steps", type=int, default=1500)
    ap.add_argument("--difficulty", type=int, default=2)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)

    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=5e-4) # Slightly lower for LSTM stability
    ap.add_argument("--batch", type=int, default=32) # Number of sequences
    ap.add_argument("--seq_len", type=int, default=10)
    ap.add_argument("--replay_episodes", type=int, default=2000)
    ap.add_argument("--warmup_episodes", type=int, default=50)
    ap.add_argument("--target_sync", type=int, default=1000)
    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.01)
    ap.add_argument("--eps_decay_steps", type=int, default=1000000) # Extended for 2k episodes
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OBELIX = import_obelix(args.obelix_py)

    q_net = DRQN()
    t_net = DRQN()
    t_net.load_state_dict(q_net.state_dict())
    t_net.eval()

    opt = optim.Adam(q_net.parameters(), lr=args.lr)
    replay = SequenceReplay(args.replay_episodes)
    steps = 0

    def get_eps(t):
        if t >= args.eps_decay_steps: return args.eps_end
        return args.eps_start + (t / args.eps_decay_steps) * (args.eps_end - args.eps_start)
    print("Starting training...")
    for ep in range(args.episodes):
        print(f"Running episode: {ep+1}")
        raw_env = OBELIX(
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty,
            box_speed=args.box_speed,
            seed=args.seed + ep,
        )
        env = RewardWrapper(raw_env)
        s = env.reset(seed=args.seed + ep)
        
        # Reset hidden state for new episode
        h = q_net.init_hidden(1)
        ep_ret = 0.0
        steps_since_signal = 0

        for _ in range(args.max_steps):
            # Check if any sensor is active
            any_signal = np.any(s > 0)
            
            if not any_signal:
                steps_since_signal += 1
                # Gradual Search Bias: FW probability ramps up by 4% each step
                fw_prob = min(0.8, 0.2 + 0.04 * steps_since_signal)
                other_prob = (1.0 - fw_prob) / 4.0
                p = [other_prob, other_prob, fw_prob, other_prob, other_prob]
                a = np.random.choice(len(ACTIONS), p=p)
                
                # Feed observation into LSTM to keep temporal context
                with torch.no_grad():
                    _, h = q_net(torch.tensor(s, dtype=torch.float32).view(1, 1, -1), h)
            else:
                steps_since_signal = 0
                eps = get_eps(steps)
                if np.random.rand() < eps:
                    a = np.random.randint(len(ACTIONS))
                    # Update hidden state even on random actions to keep context
                    with torch.no_grad():
                        _, h = q_net(torch.tensor(s, dtype=torch.float32).view(1, 1, -1), h)
                else:
                    with torch.no_grad():
                        qs, h = q_net(torch.tensor(s, dtype=torch.float32).view(1, 1, -1), h)
                    a = int(torch.argmax(qs).item())

            s2, r, done = env.step(ACTIONS[a], render=False)
            ep_ret += r
            replay.add_step(Transition(s, a, r, s2, done))
            s = s2
            steps += 1

            # Update loop
            if len(replay) >= args.warmup_episodes:
                sb, ab, rb, s2b, db = replay.sample(args.batch, args.seq_len)
                
                # Double DQN logic for sequences
                # next_q from online net to pick actions
                # next_v from target net for values
                with torch.no_grad():
                    q_next, _ = q_net(s2b) # (B, L, 5)
                    a_next = torch.argmax(q_next, dim=2, keepdim=True) # (B, L, 1)
                    
                    q_tgt, _ = t_net(s2b) # (B, L, 5)
                    v_next = q_tgt.gather(2, a_next).squeeze(2) # (B, L)
                    
                    target = rb + args.gamma * (1.0 - db) * v_next

                current_q, _ = q_net(sb) # (B, L, 5)
                current_val = current_q.gather(2, ab.unsqueeze(2)).squeeze(2) # (B, L)
                
                loss = nn.functional.smooth_l1_loss(current_val, target)
                
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                opt.step()

                if steps % args.target_sync == 0:
                    t_net.load_state_dict(q_net.state_dict())

            if done: break

        if (ep + 1) % 20 == 0:
            print(f"Ep {ep+1}/{args.episodes} Ret: {ep_ret:.1f} Eps: {get_eps(steps):.3f} Buffer: {len(replay)}")
        
        if (ep + 1) % 200 == 0:
            torch.save(q_net.state_dict(), args.out)

    torch.save(q_net.state_dict(), args.out)
    print(f"Training finished. Weights saved to {args.out}")

if __name__ == "__main__":
    main()
