import argparse
import numpy as np
import torch
import torch.nn as nn
import cv2

from obelix import OBELIX

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

# =========================
# SAME NETWORK AS TRAINING
# =========================
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


# =========================
# SENSOR CHECK (for debug)
# =========================
def sensors_active(s):
    near_right = s[[0, 2]]
    near_front = s[[4, 6, 8, 10]]
    near_left  = s[[12, 14]]
    ir = s[16]
    return (np.sum(near_right) +
            np.sum(near_front) +
            np.sum(near_left) +
            ir) > 0


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="weights.pth")
    parser.add_argument("--difficulty", type=int, default=3)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--max_steps", type=int, default=2000)
    args = parser.parse_args()

    # ===== ENV =====
    env = OBELIX(
        scaling_factor=5,
        arena_size=500,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=args.difficulty,
        box_speed=2,
    )

    # ===== LOAD MODEL =====
    model = PPO_LSTM()
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    # ===== INIT =====
    state = env.reset()
    total_reward = 0

    # LSTM hidden state
    hx = torch.zeros(1, 1, 64)
    cx = torch.zeros(1, 1, 64)

    print("Running PPO-LSTM agent... Press 'q' to quit.")

    for step in range(1, args.max_steps + 1):

        env.render_frame()

        # ===== FORWARD =====
        x = torch.tensor(state, dtype=torch.float32).view(1, 1, -1)

        with torch.no_grad():
            logits, hx, cx = model(x, hx, cx)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0, 0]

        action = int(np.argmax(probs))

        # ===== STEP =====
        state, reward, done = env.step(ACTIONS[action])
        total_reward += reward

        print(f"Step {step} | Action {ACTIONS[action]} | Reward {reward:.2f} | Total {total_reward:.2f}")

        key = cv2.waitKey(50)
        if key == ord('q'):
            break

        if done:
            print("Episode done. Total score:", total_reward)
            break

    cv2.waitKey(0)