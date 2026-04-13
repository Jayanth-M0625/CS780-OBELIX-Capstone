import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

RESULTS_FILE = "evaluation_results.csv"
OUTPUT_DIR = "evaluation_plots"

def plot_results():
    if not os.path.exists(RESULTS_FILE):
        print("No results file found.")
        return
        
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    df = pd.read_csv(RESULTS_FILE)
    
    # Levels available
    levels = df['level'].unique()
    
    for level in levels:
        level_df = df[df['level'] == level]
        
        # 1. Success Rate
        plt.figure(figsize=(10, 6))
        sr_df = level_df.groupby('agent_name')['success'].mean().reset_index()
        sns.barplot(data=sr_df, x='agent_name', y='success', palette='viridis')
        plt.title(f'Success Rate - {level}')
        plt.ylabel('Mean Success Rate')
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'success_rate_{level}.png'))
        plt.close()
        
        # 2. Reward Distribution
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=level_df, x='agent_name', y='reward', palette='Set2')
        plt.title(f'Reward Distribution - {level}')
        plt.ylabel('Total Reward')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'reward_dist_{level}.png'))
        plt.close()

    # 3. Comparative plot for Success Rate across all agents and all levels
    plt.figure(figsize=(14, 8))
    sr_comp = df.groupby(['level', 'agent_name'])['success'].mean().reset_index()
    
    # Order levels logically
    level_order = ['Level_1', 'Level_2', 'Level_3_NoWalls', 'Level_3_Walls']
    sr_comp['level'] = pd.Categorical(sr_comp['level'], categories=level_order, ordered=True)
    sr_comp = sr_comp.sort_values('level')

    sns.barplot(data=sr_comp, x='level', y='success', hue='agent_name')
    plt.title('Success Rate Comparison - All Agents & Levels')
    plt.ylabel('Mean Success Rate')
    plt.ylim(0, 1.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'success_rate_cross_level.png'))
    plt.savefig(os.path.join(OUTPUT_DIR, 'success_rate_cross_level.svg'))
    plt.close()

    print(f"Plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    plot_results()
