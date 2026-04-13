import os
import shutil
import importlib.util
import csv
import numpy as np
import torch
from obelix import OBELIX
from tqdm import tqdm

W_DIR = "All experiments weights"
TEMP_DIR = "eval_temp"
RESULTS_FILE = "evaluation_results.csv"

# Configuration settings
MAX_STEPS = 1000
SCALING_FACTOR = 5
ARENA_SIZE = 500

BIAS_VALUES = [0.4, 0.6]

# Levels with their params: (difficulty, wall_obstacles, runs, level_name)
LEVELS_TO_RUN = [
    (0, False, 10, "Level_1"),
    (3, False, 20, "Level_3_NoWalls"),
    (3, True, 20, "Level_3_Walls"),
]

def run_episode(env, agent_policy, seed):
    obs = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    total_reward = 0
    steps = 0
    collisions = 0
    prev_stuck = 0
    
    done = False
    while not done and steps < MAX_STEPS:
        action = agent_policy(obs, rng)
        obs, reward, done = env.step(action, render=False)
        total_reward += float(reward)
        steps += 1
        
        current_stuck = int(obs[17]) # stuck_flag
        if current_stuck == 1 and prev_stuck == 0:
            collisions += 1
        prev_stuck = current_stuck
        
    success = 1 if total_reward >= 1500 else 0 
    return {"reward": total_reward, "steps": steps, "collisions": collisions, "success": success}

def evaluate_set(agent_file, weight_file, display_base_name, bias_val, difficulty, obstacles, runs, level_name, writer, f):
    display_name = f"{display_base_name}_{bias_val}"
    print(f"\n--- Bias Sweep: {level_name} (Bias {bias_val}, {runs} runs) ---")
    
    # Set bias env var
    os.environ["OBELIX_BIAS"] = str(bias_val)
    
    # Prepare temp agent
    agent_path = os.path.join(W_DIR, agent_file)
    weight_path = os.path.join(W_DIR, weight_file)
    shutil.copy(agent_path, os.path.join(TEMP_DIR, "agent.py"))
    shutil.copy(weight_path, os.path.join(TEMP_DIR, "weights.pth"))
    
    # Load agent
    spec = importlib.util.spec_from_file_location("eval_agent", os.path.join(TEMP_DIR, "agent.py"))
    agent_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_module)
    
    env = OBELIX(
        scaling_factor=SCALING_FACTOR,
        arena_size=ARENA_SIZE,
        max_steps=MAX_STEPS,
        wall_obstacles=obstacles,
        difficulty=difficulty,
        seed=0
    )
    
    # Higher seeds to keep them distinct from previous CURIOUS tests if possible
    # We'll use 20-39 for these runs
    for run_id in tqdm(range(20, 20 + runs), desc="    Runs"):
        res = run_episode(env, agent_module.policy, seed=run_id)
        writer.writerow([
            level_name,
            display_name,
            weight_file,
            difficulty,
            int(obstacles),
            run_id,
            res["reward"],
            res["steps"],
            res["collisions"],
            res["success"]
        ])
        f.flush()

def main():
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
        
    # STRICT APPEND mode
    f = open(RESULTS_FILE, "a", newline="")
    writer = csv.writer(f)
    
    # Run the comprehensive loop
    for diff, obstacles, runs, level_name in LEVELS_TO_RUN:
        for b in BIAS_VALUES:
            evaluate_set(
                "agent_d3qnBIASED.py", 
                "weights_d3qn_RewSh2.pth", 
                "D3QN_RewSh2_BIASED", 
                b, 
                difficulty=diff, 
                obstacles=obstacles, 
                runs=runs, 
                level_name=level_name, 
                writer=writer, 
                f=f
            )
    
    f.close()
    print("\nFinal bias sweep sequence complete. Results stored in", RESULTS_FILE)

if __name__ == "__main__":
    main()
