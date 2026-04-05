import argparse
import importlib.util
import numpy as np
import matplotlib.pyplot as plt

# usage: python metrics.py --obelix obelix.py --agent agent.py --difficulty 0

# =========================================================
# Load OBELIX dynamically
# =========================================================
def load_env(obelix_path):
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OBELIX


# =========================================================
# Load agent policy dynamically
# =========================================================
def load_policy(agent_path, weights_path):
    spec = importlib.util.spec_from_file_location("agent_module", agent_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # CASE 1: clean agent with load()
    if hasattr(module, "load"):
        module.load(weights_path)
        return module.policy

    # CASE 2: submission-style agent (no load function)
    else:
        import os
        import shutil

        # Copy weights to expected filename
        agent_dir = os.path.dirname(agent_path)
        target_path = os.path.join(agent_dir, "weights.pth")

        if weights_path != target_path:
            shutil.copy(weights_path, target_path)

        return module.policy


# =========================================================
# Evaluation with metrics
# =========================================================
def evaluate_with_metrics(env_class, policy_fn, args):
    results = {
        "episode_rewards": [],
        "success": [],
        "collisions": [],
        "lengths": []
    }

    for i in range(args.runs):
        print(f"Running run: {i+1}")
        env = env_class(
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty,
            box_speed=args.box_speed,
            seed=i,
        )

        obs = env.reset(seed=i)
        rng = np.random.default_rng(i)

        total_reward = 0
        steps = 0
        collisions = 0

        done = False

        while not done:
            action = policy_fn(obs, rng)
            obs, reward, done = env.step(action, render=False)

            total_reward += reward
            steps += 1

            if obs[17] == 1:
                collisions += 1

        success = int(env.enable_push and done)

        results["episode_rewards"].append(total_reward)
        results["success"].append(success)
        results["collisions"].append(collisions)
        results["lengths"].append(steps)

    return results


# =========================================================
# Plot
# =========================================================
def plot_metrics(results):
    rewards = results["episode_rewards"]
    success = results["success"]
    collisions = results["collisions"]
    lengths = results["lengths"]

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    axs[0, 0].plot(rewards)
    axs[0, 0].set_title("Episode Reward")

    axs[0, 1].plot(np.cumsum(success) / (np.arange(len(success)) + 1))
    axs[0, 1].set_title("Success Rate")

    axs[1, 0].plot(collisions)
    axs[1, 0].set_title("Collision Count")

    axs[1, 1].plot(lengths)
    axs[1, 1].set_title("Episode Length")

    plt.tight_layout()
    plt.show()


# =========================================================
# MAIN (CLI)
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--obelix", type=str, required=True)
    parser.add_argument("--agent", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--difficulty", type=int, default=3)
    parser.add_argument("--wall_obstacles", action="store_true")

    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--box_speed", type=int, default=2)
    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)

    args = parser.parse_args()

    # Load dynamically
    OBELIX = load_env(args.obelix)
    policy_fn = load_policy(args.agent, args.weights)

    # Evaluate
    results = evaluate_with_metrics(OBELIX, policy_fn, args)

    # Print summary
    print("\n===== SUMMARY =====")
    print("Avg Reward:", np.mean(results["episode_rewards"]))
    print("Success Rate:", np.mean(results["success"]))
    print("Avg Collisions:", np.mean(results["collisions"]))
    print("Avg Length:", np.mean(results["lengths"]))

    # Plot
    plot_metrics(results)