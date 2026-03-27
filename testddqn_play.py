import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn

from obelix import OBELIX

# usage: python testd3qn_play.py --weights checkpoint_200.pth --difficulty 2

# ===== SAME NETWORK AS TRAINING =====
class DQN(nn.Module):
    def __init__(self, in_dim=18, n_actions=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )
    def forward(self, x):
        return self.net(x)


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
    q = DQN()
    q.load_state_dict(torch.load(args.weights, map_location="cpu"))
    q.eval()

    ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

    state = bot.reset()
    episode_reward = 0

    print("Running trained agent... Press 'q' to quit.")

    for step in range(1, args.max_steps + 1):
        bot.render_frame()

        # ===== MODEL ACTION =====
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