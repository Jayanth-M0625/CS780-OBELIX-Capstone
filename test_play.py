import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import os

from obelix import OBELIX

# usage: python test_play.py --weights weights_d3qn_RewSh2.pth --difficulty 3 --wall_obstacles

# ===== SAME NETWORK AS TRAINING =====
class D3QN(nn.Module):
    def __init__(self, in_dim=18, n_actions=5):
        super().__init__()
        # shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
        )
        # value stream
        self.value = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
    def forward(self, x):
        f = self.feature(x)
        v = self.value(f)                 
        a = self.advantage(f)             
        # combine streams
        q = v + a - a.mean(dim=1, keepdim=True)
        return q

# ===== SENSOR CHECK FUNCTION =====
def sensors_active(s):
    near_right = s[[0, 2]]
    near_front = s[[4, 6, 8, 10]]
    near_left  = s[[12, 14]]
    ir = s[16]
    return (np.sum(near_right) +
            np.sum(near_front) +
            np.sum(near_left) +
            ir) > 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Defaulting to the best weights for Level 3
    parser.add_argument("--weights", type=str, default="weights_d3qn_RewSh2.pth")
    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--difficulty", type=int, default=3)
    parser.add_argument("--box_speed", type=int, default=2)
    parser.add_argument("--bias", type=float, default=0.8)
    args = parser.parse_args()

    # ===== ENV =====
    bot = OBELIX(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
    )

    # ===== LOAD MODEL =====
    print(f"Loading weights from {args.weights}...")
    q = D3QN()
    state_dict = torch.load(args.weights, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    q.load_state_dict(state_dict)
    q.eval()

    ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
    rng = np.random.default_rng()

    state = bot.reset()
    episode_reward = 0

    print("Running D3QN agent with Biased Random Policy... Press 'q' to quit.")
    print(f"Biased Random Policy: {args.bias} FW when sensors are inactive.")

    for step in range(1, args.max_steps + 1):
        bot.render_frame()

        # ===== ACTION =====
        if not sensors_active(state):
            # Biased random policy
            other_p = (1.0 - args.bias) / 4.0
            p = [other_p, other_p, args.bias, other_p, other_p]
            action_idx = rng.choice(len(ACTIONS), p=p)
        else:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_vals = q(s).squeeze(0).numpy()
            action_idx = int(np.argmax(q_vals))

        # ===== STEP =====
        state, reward, done = bot.step(ACTIONS[action_idx])
        episode_reward += reward

        if step % 10 == 0:
            print(f"Step {step} | Action {ACTIONS[action_idx]} | Reward {reward:.2f} | Total {episode_reward:.2f}")

        # Visualization delay
        key = cv2.waitKey(20)

        if key == ord('q'):
            break

        if done:
            print("Episode done. Total score:", episode_reward)
            break

    # Wait for key press before closing
    print("Finished. Press any key in the window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
