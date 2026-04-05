print("Importing libraries...")
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from reward_wrapper import RewardWrapper

# usage: python train_ppoLSTM.py --obelix_py obelix.py --difficulty 3 --wall_obstacles

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
print("Initializing classes...")
# =========================
# PPO + LSTM Network
# =========================
class PPO_LSTM(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=64, n_actions=5):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.policy = nn.Linear(hidden_dim, n_actions)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x, hx, cx):
        x = torch.tanh(self.fc(x))
        x, (hx, cx) = self.lstm(x, (hx, cx))
        logits = self.policy(x)
        value = self.value(x)
        return logits, value, hx, cx


# =========================
# GAE
# =========================
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    adv, gae = [], 0
    values = values + [0]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        adv.insert(0, gae)

    returns = [a + v for a, v in zip(adv, values[:-1])]
    return adv, returns


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

    # load env
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", args.obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    OBELIX = mod.OBELIX

    model = PPO_LSTM()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    print("Starting training...")
    for ep in range(args.episodes):
        print(f"Running episode: {ep+1}")
        env = RewardWrapper(OBELIX(
            scaling_factor=5,
            arena_size=500,
            max_steps=1000,
            wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty,
            seed=ep
        ))
        state = env.reset(seed=ep)
        hx = torch.zeros(1, 1, 64)
        cx = torch.zeros(1, 1, 64)
        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
        for t in range(1000):
            s = torch.tensor(state, dtype=torch.float32).view(1,1,-1)
            logits, value, hx, cx = model(s, hx, cx)
            logits = logits.squeeze(0)
            # ===== Forward Bias (SAFE) =====
            near_right = state[[0,2]]
            near_front = state[[4,6,8,10]]
            near_left  = state[[12,14]]
            ir = state[16]
            any_signal = (np.sum(near_right)+np.sum(near_front)+np.sum(near_left)+ir) > 0
            if ep<200 and not any_signal:
                logits[0,2] += 0.2   # bias FW action intially for exploration
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            logp = dist.log_prob(action)
            next_state, reward, done = env.step(ACTIONS[action.item()], render=False)
            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            dones.append(done)
            log_probs.append(logp.item())
            values.append(value.item())
            state = next_state
            if done:
                break
        adv, returns = compute_gae(rewards, values, dones)
        states = torch.tensor(np.array(states), dtype=torch.float32).view(len(states),1,-1)
        actions = torch.tensor(actions)
        old_log_probs = torch.tensor(log_probs)
        returns = torch.tensor(returns, dtype=torch.float32)
        adv = torch.tensor(adv, dtype=torch.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        hx = torch.zeros(1, len(states), 64)
        cx = torch.zeros(1, len(states), 64)
        logits, values_pred, _, _ = model(states, hx, cx)
        logits = logits.squeeze(1)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        new_log_probs = dist.log_prob(actions)
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 0.8, 1.2) * adv
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = ((returns - values_pred.squeeze())**2).mean()
        entropy = dist.entropy().mean()
        beta = max(0.01 * (1 - ep / 1500), 0.001) # decaying beta
        loss = policy_loss + 0.5 * value_loss - beta * entropy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (ep+1) % 50 == 0:
            print(f"Episode {ep+1}, Reward: {sum(rewards):.2f}")
        if (ep+1) % 200 == 0:
            torch.save(model.state_dict(), f"ppo_lstm_{ep+1}.pth")
    torch.save(model.state_dict(), "weights.pth")


if __name__ == "__main__":
    main()