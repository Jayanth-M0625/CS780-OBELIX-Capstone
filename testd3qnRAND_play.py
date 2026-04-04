import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn

from obelix import OBELIX

# usage: python testd3qn_play.py --weights checkpoint_200.pth --difficulty 2

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
    parser.add_argument("--weights", type=str, default="weights.pth")
    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--difficulty", type=int, default=0)
    parser.add_argument("--box_speed", type=int, default=2)
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
    q = D3QN()
    q.load_state_dict(torch.load(args.weights, map_location="cpu"))
    q.eval()

    ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

    state = bot.reset()
    episode_reward = 0

    print("Running trained agent... Press 'q' to quit.")

    for step in range(1, args.max_steps + 1):
        bot.render_frame()

        # ===== ACTION =====
        if not sensors_active(state):
            action = np.random.randint(len(ACTIONS))
        else:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_vals = q(s).squeeze(0).numpy()
            action = int(np.argmax(q_vals))

        # ===== STEP =====
        state, reward, done = bot.step(ACTIONS[action])
        episode_reward += reward

        print(f"Step {step} | Action {ACTIONS[action]} | Reward {reward:.2f} | Total {episode_reward:.2f}")

        # slow down so you can see
        key = cv2.waitKey(50)

        if key == ord('q'):
            break

        if done:
            print("Episode done. Total score:", episode_reward)
            break

    cv2.waitKey(0)