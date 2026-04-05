print("Importing libraries...")
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from reward_wrapper import RewardWrapper


ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
print("Initializing classes...")
# =========================
# MODEL
# =========================
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
        return self.policy(x), self.value(x)


# =========================
# TRAIN
# =========================
print("Starting main training...")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obelix_py", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--difficulty", type=int, default=3)
    parser.add_argument("--wall_obstacles", action="store_true")
    args = parser.parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", args.obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    OBELIX = mod.OBELIX

    model = VPG()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    beta_start = 0.01
    beta_end = 0.001
    gamma = 0.99
    print("Starting training...")
    for ep in range(args.episodes):
        print(f"Running episode: {ep+1}")
        base_env = OBELIX(
            scaling_factor=5,
            arena_size=500,
            max_steps=1000,
            wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty,
            seed=ep
        )
        env = RewardWrapper(base_env)
        state = env.reset(seed=ep)

        states, actions, rewards, log_probs, values = [], [], [], [], []

        for t in range(1000):
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            logits, value = model(s)
            probs = torch.softmax(logits, dim=-1)

            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            next_state, reward, done = env.step(ACTIONS[action.item()], render=False)

            states.append(s)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(dist.log_prob(action))
            values.append(value.squeeze())

            state = next_state
            if done:
                break

        # =========================
        # RETURNS
        # =========================
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.stack(values)

        advantages = returns - values.detach()

        log_probs = torch.stack(log_probs)

        # entropy
        logits, _ = model(torch.cat(states))
        probs = torch.softmax(logits, dim=-1)
        entropy = torch.distributions.Categorical(probs).entropy().mean()

        # =========================
        # LOSS
        # =========================
        policy_loss = -(log_probs * advantages).mean()
        value_loss = ((returns - values)**2).mean()
        beta = max(beta_end, beta_start * (1 - ep / args.episodes)) # decaying beta
        loss = policy_loss + 0.5 * value_loss - beta * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}, Reward: {sum(rewards):.2f}")

        if (ep + 1) % 200 == 0:
            torch.save(model.state_dict(), f"vpg_{ep+1}.pth")

    torch.save(model.state_dict(), "vpg_weights.pth")


if __name__ == "__main__":
    main()