print("Importing libraries...")
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from reward_wrapper import RewardWrapper

# usage : python train_ppo.py --obelix_py obelix.py --difficulty 2 --wall_obstacles

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
print("Initializing classes...")
# =========================
# Policy + Value Network
# =========================
class PPOAgent(nn.Module):
    def __init__(self, obs_dim=18, n_actions=5):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
        )

        self.policy = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.shared(x)
        logits = self.policy(x)
        value = self.value(x)
        return logits, value


# =========================
# Utility
# =========================
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    adv = []
    gae = 0
    values = values + [0]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        adv.insert(0, gae)

    returns = [a + v for a, v in zip(adv, values[:-1])]
    return adv, returns


# =========================
# Main Training
# =========================
print("Starting main training...")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obelix_py", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--difficulty", type=int, default=3)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    # load env dynamically
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", args.obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    OBELIX = mod.OBELIX

    agent = PPOAgent()
    optimizer = optim.Adam(agent.parameters(), lr=args.lr)
    best_reward = -1e9
    print("Starting training...")
    for ep in range(args.episodes):
        print(f"Running episode: {ep+1}")
        env = RewardWrapper(OBELIX(
            scaling_factor=5,
            arena_size=500,
            max_steps=args.max_steps,
            wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty,
            box_speed=2,
            seed=ep
        ))

        s = env.reset(seed=ep)

        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

        for t in range(args.max_steps):

            s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            logits, value = agent(s_t)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)

            a = dist.sample()
            logp = dist.log_prob(a)

            s2, r, done = env.step(ACTIONS[a.item()], render=False)

            states.append(s)
            actions.append(a.item())
            rewards.append(r)
            dones.append(done)
            log_probs.append(logp.item())
            values.append(value.item())

            s = s2
            if done:
                break

        # ===== GAE =====
        adv, returns = compute_gae(rewards, values, dones)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions)
        old_log_probs = torch.tensor(log_probs)
        returns = torch.tensor(returns, dtype=torch.float32)
        adv = torch.tensor(adv, dtype=torch.float32)

        # normalize advantage
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        # ===== PPO UPDATE =====
        for _ in range(args.epochs):
            logits, values_pred = agent(states)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)

            new_log_probs = dist.log_prob(actions)

            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = ((returns - values_pred.squeeze()) ** 2).mean()

            loss = policy_loss + 0.5 * value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ep_reward = sum(rewards)

        if ep_reward > best_reward:
            best_reward = ep_reward
            torch.save(agent.state_dict(), "ppo_best.pth")
        if (ep+1) % 50 == 0:
            print(f"Episode {ep+1} reward: {sum(rewards):.2f}")

        if (ep + 1) % 200 == 0:
            torch.save(agent.state_dict(), f"ppo_checkpoint_{ep+1}.pth")
            print(f"Saved checkpoint at episode {ep+1}")

    torch.save(agent.state_dict(), "ppo_weights.pth")
    print("Saved ppo_weights.pth")


if __name__ == "__main__":
    main()