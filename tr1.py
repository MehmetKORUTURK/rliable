import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List
import seaborn as sns

def load_rewards(paths: List[Path]) -> List[np.ndarray]:
    reward_lists = []

    for path in paths:
        try:
            df = pd.read_csv(path, comment='#')
        except Exception as e:
            print(f"Failed to read {path}: {e}")
            continue

        reward_col = next((col for col in df.columns if col.lower() in ['r', 'reward']), None)
        if reward_col is None:
            print(f"Warning: Reward column not found in {path}. Skipping.")
            continue

        rewards = pd.to_numeric(df[reward_col], errors='coerce').dropna().values
        if rewards.size == 0:
            print(f"Warning: No valid reward data in {path}. Skipping.")
            continue

        reward_lists.append(rewards)

    return reward_lists


def compute_distribution_with_ci(data: np.ndarray, taus: np.ndarray, n_bootstrap: int = 1000, ci: float = 95):
    """
    Bootstrap confidence interval over run-score distribution for a single run.
    """
    distributions = []

    for _ in range(n_bootstrap):
        sampled = np.random.choice(data, size=len(data), replace=True)
        dist = [(sampled > tau).mean() for tau in taus]
        distributions.append(dist)

    distributions = np.array(distributions)
    lower = np.percentile(distributions, (100 - ci) / 2, axis=0)
    upper = np.percentile(distributions, 100 - (100 - ci) / 2, axis=0)
    mean = np.mean(distributions, axis=0)

    return mean, lower, upper


def plot_with_confidence(reward_data: List[np.ndarray], labels: List[str], title: str = "Run-Score Distribution with 95% CI"):
    all_scores = np.concatenate(reward_data)
    taus = np.linspace(np.min(all_scores), np.max(all_scores), 100)

    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("Set2", len(reward_data))

    for idx, (rewards, label) in enumerate(zip(reward_data, labels)):
        mean, lower, upper = compute_distribution_with_ci(rewards, taus)
        plt.plot(taus, mean, label=label, color=colors[idx])
        plt.fill_between(taus, lower, upper, alpha=0.3, color=colors[idx])

    plt.axhline(0.5, color='gray', linestyle='--', label="Median (y=0.5)")
    plt.xlabel("Human Normalized Score (τ)")
    plt.ylabel("Fraction of runs with score > τ")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    csv_paths = [
        Path(r"C:\Users\mkoru\OneDrive\Desktop\programs\sustaingym-main\sustaingym-main\sustaingym\logs\building_PPO\2025-03-20-07-47-08_DS_0.0_DA_0.0\monitor.csv"),
        Path(r"C:\Users\mkoru\OneDrive\Desktop\programs\sustaingym-main\sustaingym-main\sustaingym\logs\building_PPO\2025-03-23-15-13-01_DS_0.2_DA_0.0\monitor.csv"),
        Path(r"C:\Users\mkoru\OneDrive\Desktop\programs\sustaingym-main\sustaingym-main\sustaingym\logs\building_PPO\2025-03-23-15-13-31_DS_0.3_DA_0.0\monitor.csv"),
    ]

    labels = [
        "Algorithm A (DS=0.0)",
        "Algorithm B (DS=0.2)",
        "Algorithm C (DS=0.3)",
    ]

    reward_data = load_rewards(csv_paths)
    plot_with_confidence(reward_data, labels)


if __name__ == "__main__":
    main()
