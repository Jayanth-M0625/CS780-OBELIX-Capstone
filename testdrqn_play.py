"""Visualization script for DRQN agent in OBELIX environment.

Usage:
  python testdrqp_play.py --weights weights_drqn.pth --difficulty 2
"""

import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn

from obelix import OBELIX

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="weights_drqn.pth")
    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--difficulty", type=int, default=2)
    parser.add_argument("--box_speed", type=int, default=2)
    args = parser.parse_args()

    # Create environment
    bot = OBELIX(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
    )

    # Load Model
    model = DRQN()
    if torch.cuda.is_available():
        model = model.cuda()
    
    try:
        sd = torch.load(args.weights, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd)
        print(f"Loaded weights from {args.weights}")
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Continuing with random weights for visualization Demo.")

    model.eval()

    ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
    
    state = bot.reset()
    hidden = model.init_hidden(1)
    if torch.cuda.is_available():
        hidden = (hidden[0].cuda(), hidden[1].cuda())

    episode_reward = 0
    steps_since_signal = 0
    print("Running DRQN agent... Press 'q' to quit.")

    for step in range(1, args.max_steps + 1):
        bot.render_frame()

        # Check signal
        any_signal = np.any(state > 0)
        
        if not any_signal:
            steps_since_signal += 1
            # Gradual Search Bias
            fw_prob = min(0.8, 0.2 + 0.04 * steps_since_signal)
            other_prob = (1.0 - fw_prob) / 4.0
            p = [other_prob, other_prob, fw_prob, other_prob, other_prob]
            action_idx = int(np.random.choice(len(ACTIONS), p=p))
            
            # Feed to LSTM
            with torch.no_grad():
                s_tensor = torch.tensor(state, dtype=torch.float32).view(1, 1, -1)
                if torch.cuda.is_available():
                    s_tensor = s_tensor.cuda()
                _, hidden = model(s_tensor, hidden)
        else:
            steps_since_signal = 0
            with torch.no_grad():
                s_tensor = torch.tensor(state, dtype=torch.float32).view(1, 1, -1)
                if torch.cuda.is_available():
                    s_tensor = s_tensor.cuda()
                
                q_vals, hidden = model(s_tensor, hidden)
                action_idx = int(torch.argmax(q_vals).item())
        
        action_name = ACTIONS[action_idx]
        state, reward, done = bot.step(action_name)
        episode_reward += reward

        print(f"Step {step} | Action {action_name} | Reward {reward:.2f} | Total {episode_reward:.2f}", end="\r")

        key = cv2.waitKey(20)
        if key == ord('q'):
            break
        if done:
            print(f"\nEpisode done. Total score: {episode_reward:.2f}")
            break

    cv2.destroyAllWindows()
