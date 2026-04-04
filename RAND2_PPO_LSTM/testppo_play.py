import argparse
import numpy as np
import torch
from obelix import OBELIX
from agent_ppo import PPOAgent
import torch.nn as nn

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

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

        self.value = nn.Sequential(   # ADD THIS
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.shared(x)
        return self.policy(x)  # ignore value at inference
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="ppo_weights.pth")
    parser.add_argument("--difficulty", type=int, default=3)
    parser.add_argument("--wall_obstacles", action="store_true")
    args = parser.parse_args()

    env = OBELIX(
        scaling_factor=5,
        arena_size=500,
        max_steps=2000,
        wall_obstacles=args.wall_obstacles,
        difficulty=args.difficulty,
        box_speed=2
    )

    model = PPOAgent()
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    state = env.reset()
    rng = np.random.default_rng(0)

    total = 0

    while True:
        env.render_frame()

        x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=-1).numpy()[0]

        action = int(rng.choice(len(ACTIONS), p=probs))

        state, reward, done = env.step(ACTIONS[action])
        total += reward

        if done:
            print("Done. Score:", total)
            break


if __name__ == "__main__":
    main()