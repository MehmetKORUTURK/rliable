from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------
def load_rewards(paths) -> list:
    reward_lists = []
    for path in paths:
        df = pd.read_csv(path, comment='#')
        reward_col = next((col for col in df.columns if col.lower() in ['r', 'reward']), None)
        if not reward_col:
            raise ValueError(f"Reward sütunu bulunamadı: {df.columns.tolist()}")
        rewards = pd.to_numeric(df[reward_col], errors='coerce').dropna().values
        reward_lists.append(rewards)
    return reward_lists

# ---------------------------------------------
def compute_metric(data: np.ndarray, metric: str) -> float:
    if metric == 'mean':
        return np.mean(data)
    elif metric == 'median':
        return np.median(data)
    elif metric == 'iqm':
        q1, q3 = np.percentile(data, [25, 75])
        return np.mean(data[(data >= q1) & (data <= q3)])
    else:
        raise ValueError(f"Bilinmeyen metrik: {metric}")

# ---------------------------------------------
def stratified_bootstrap_ci(data_list, metric='median', ci=95, n_boot=10000, plot_dist=False, group_name=""):
    sample_stats = []
    for _ in range(n_boot):
        resampled = [np.random.choice(task, size=len(task), replace=True) for task in data_list]
        stat = compute_metric(np.concatenate(resampled), metric)
        sample_stats.append(stat)

    all_data = np.concatenate(data_list)
    point_estimate = compute_metric(all_data, metric)
    lower, upper = np.percentile(sample_stats, [(100 - ci) / 2, 100 - (100 - ci) / 2])

    if plot_dist:
        plt.figure(figsize=(8, 5))
        plt.hist(sample_stats, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(point_estimate, color='blue', linestyle='--', label=f'{metric.upper()}: {point_estimate:.2f}')
        plt.axvline(lower, color='red', linestyle='--', label=f'CI Lower: {lower:.2f}')
        plt.axvline(upper, color='green', linestyle='--', label=f'CI Upper: {upper:.2f}')
        plt.title(f'{group_name} - {metric.upper()} Bootstrap Dağılımı')
        plt.xlabel(f'{metric.upper()} Değeri')
        plt.ylabel('Frekans')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return point_estimate, lower, upper

# ---------------------------------------------
def plot_dual_confidence_intervals(result_dict):
    groups = list(result_dict.keys())
    y_pos = np.arange(len(groups))
    fig, axs = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
    metrics = ['median', 'iqm']

    for i, metric in enumerate(metrics):
        means = [result_dict[g][metric][0] for g in groups]
        lowers = [means[j] - result_dict[g][metric][1] for j, g in enumerate(groups)]
        uppers = [result_dict[g][metric][2] - means[j] for j, g in enumerate(groups)]

        axs[i].errorbar(means, y_pos, xerr=[lowers, uppers], fmt='o', capsize=5,
                        linestyle='none', label=metric.upper())
        axs[i].set_title(metric.upper())
        axs[i].set_xlabel('95% Confidence Interval')
        axs[i].grid(True, axis='x', linestyle='--', alpha=0.5)

    axs[0].set_yticks(y_pos)
    axs[0].set_yticklabels(groups)
    axs[1].set_yticks([])
    plt.suptitle('Stratified Bootstrap Confidence Intervals')
    plt.tight_layout()
    plt.show()

# ---------------------------------------------
# 🛠️ KULLANIM: Dosya yollarını ve run gruplarını tanımla

run_groups = {
    "3 runs": [
        Path(r"C:\Users\mkoru\OneDrive\Desktop\programs\sustaingym-main\sustaingym-main\sustaingym\logs\building_PPO\2025-03-20-07-47-08_DS_0.0_DA_0.0\monitor.csv"),
        Path(r"C:\Users\mkoru\OneDrive\Desktop\programs\sustaingym-main\sustaingym-main\sustaingym\logs\building_PPO\2025-03-23-15-13-01_DS_0.2_DA_0.0\monitor.csv"),
        Path(r"C:\Users\mkoru\OneDrive\Desktop\programs\sustaingym-main\sustaingym-main\sustaingym\logs\building_PPO\2025-03-23-15-13-31_DS_0.3_DA_0.0\monitor.csv"),
    ],
    "5 runs": [
        Path(r"C:\Users\mkoru\OneDrive\Desktop\programs\sustaingym-main\sustaingym-main\sustaingym\logs\building_PPO\2025-03-20-07-47-08_DS_0.0_DA_0.0\monitor.csv"),
        Path(r"C:\Users\mkoru\OneDrive\Desktop\programs\sustaingym-main\sustaingym-main\sustaingym\logs\building_PPO\2025-03-23-15-13-01_DS_0.2_DA_0.0\monitor.csv"),
        Path(r"C:\Users\mkoru\OneDrive\Desktop\programs\sustaingym-main\sustaingym-main\sustaingym\logs\building_PPO\2025-03-23-15-13-31_DS_0.3_DA_0.0\monitor.csv"),
        Path(r"C:\Users\mkoru\OneDrive\Desktop\programs\sustaingym-main\sustaingym-main\sustaingym\logs\building_PPO\2025-03-23-15-14-10_DS_0.4_DA_0.0\monitor.csv"),
        Path(r"C:\Users\mkoru\OneDrive\Desktop\programs\sustaingym-main\sustaingym-main\sustaingym\logs\building_PPO\2025-03-23-15-14-36_DS_0.5_DA_0.0\monitor.csv"),
    ],
    # Devam edebilir: "10 runs", "25 runs", ...
}

all_results = {}

for group_name, paths in run_groups.items():
    try:
        rewards = load_rewards(paths)
        all_results[group_name] = {}
        for metric in ['median', 'iqm']:
            mean, low, high = stratified_bootstrap_ci(
                rewards,
                metric=metric,
                plot_dist=True,       # Histogram çizimi aktif
                group_name=group_name
            )
            all_results[group_name][metric] = (mean, low, high)
            print(f"✅ {group_name} | {metric.upper()} = {mean:.3f} [CI: {low:.3f}, {high:.3f}]")
    except Exception as e:
        print(f"⚠️ {group_name} hata: {e}")

# 🎨 Özet grafik: horizontal CI'lar
plot_dual_confidence_intervals(all_results)



