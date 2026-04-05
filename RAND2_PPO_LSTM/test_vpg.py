import argparse
import numpy as np
import torch
import torch.nn as nn
import cv2

from obelix import OBELIX

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

class VPG(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(18, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.policy = nn.Linear(64, 5)
        self.value = nn.Linear(64, 1)
    def forward(self, x):
        x = self.fc(x)
        return self.policy(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="weights.pth")
    parser.add_argument("--difficulty", type=int, default=3)
    parser.add_argument("--wall_obstacles", action="store_true")
    args = parser.parse_args()

    env = OBELIX(
        scaling_factor=5,
        arena_size=500,
        max_steps=2000,
        wall_obstacles=args.wall_obstacles,
        difficulty=args.difficulty,
        box_speed=2,
    )

    model = VPG()
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    state = env.reset()
    total_reward = 0

    print("Running VPG agent... Press 'q' to quit.")

    for step in range(2000):
        env.render_frame()

        x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        action = int(np.argmax(probs))

        state, reward, done = env.step(ACTIONS[action])
        total_reward += reward

        print(f"Step {step} | Action {ACTIONS[action]} | Reward {reward:.2f} | Total {total_reward:.2f}")

        if cv2.waitKey(50) == ord('q'):
            break

        if done:
            print("Episode done. Total score:", total_reward)
            break

    cv2.waitKey(0)