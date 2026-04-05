import os
import sys
import shutil
import importlib.util

from evaluate_on_codabench import evaluate_agent


# =========================================================
# Load agent fresh (to reset _model)
# =========================================================
def load_policy(agent_path="agent_ppoLSTM.py"):
    spec = importlib.util.spec_from_file_location("agent_module", agent_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.policy


# =========================================================
# Evaluate ONE checkpoint
# =========================================================
def evaluate_one(weights_path, agent_path="agent_ppoLSTM.py", output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    agent_dir = os.path.dirname(os.path.abspath(agent_path))
    target_weights = os.path.join(agent_dir, "weights.pth")

    fname = os.path.basename(weights_path)

    print(f"\n🔍 Evaluating: {fname}")

    # Copy → weights.pth (what agent expects)
    shutil.copy(weights_path, target_weights)

    # Reload agent (VERY IMPORTANT)
    policy = load_policy(agent_path)

    # Evaluate
    results = evaluate_agent(policy)
    score = results["mean_score"]

    # Save result
    out_file = os.path.join(output_dir, fname.replace(".pth", ".txt"))
    with open(out_file, "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v:.3f}\n")

    print(f"Score: {score:.3f}")
    return score


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_single_eval.py <weights.pth>")
        sys.exit(1)

    weights_path = sys.argv[1]
    evaluate_one(weights_path)