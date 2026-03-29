from run_single_eval import evaluate_one

# 🔴 EDIT THIS LIST however you want
checkpoints = [
    "checkpoint_200.pth",
    "checkpoint_400.pth",
    "checkpoint_600.pth",
    "checkpoint_800.pth",
    "checkpoint_1000.pth",
    "checkpoint_1200.pth",
    "checkpoint_1400.pth",
    "checkpoint_1600.pth",
    "checkpoint_1800.pth",
    "checkpoint_2000.pth",
    "weights.pth"
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