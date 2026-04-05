from run_single_eval import evaluate_one

# 🔴 EDIT THIS LIST however you want
checkpoints = [
    "ppo_lstm_600.pth",
    "ppo_lstm_800.pth",
    "ppo_lstm_1000.pth",
    "ppo_lstm_1200.pth",
    "ppo_lstm_1400.pth",
    "ppo_lstm_1600.pth",
    "ppo_lstm_1800.pth",
    "ppo_lstm_2000.pth",
    "ppo_lstm_weights.pth"
]

best_score = -1e9
best_file = None

for ckpt in checkpoints:
    score = evaluate_one(ckpt)

    if score > best_score:
        best_score = score
        best_file = ckpt

print("\n======================")
print(f"🏆 BEST: {best_file}")
print(f"🔥 SCORE: {best_score:.3f}")