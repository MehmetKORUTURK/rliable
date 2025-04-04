import numpy as np
import matplotlib.pyplot as plt
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils

# Define the algorithms list exactly as shown
algorithms = ['DQN (Nature)', 'DQN (Adam)', 'C51', 'REM', 'Rainbow', 
              'IQN', 'M-IQN', 'DreamerV2']

# Create synthetic data that resembles Atari performance scores
# Let's assume we have 10 runs and 50 games for each algorithm
num_runs = 10
num_games = 50
np.random.seed(42)  # For reproducibility

# Create synthetic data with performance characteristics that mimic the image
# Newer algorithms generally perform better
atari_200m_normalized_score_dict = {}

# DQN (Nature) - oldest algorithm, moderate performance
atari_200m_normalized_score_dict['DQN (Nature)'] = np.random.beta(2, 4, size=(num_runs, num_games)) * 1.0

# DQN (Adam) - slightly better than Nature
atari_200m_normalized_score_dict['DQN (Adam)'] = np.random.beta(2.2, 3.8, size=(num_runs, num_games)) * 1.1

# C51 - categorical DQN, better than DQN
atari_200m_normalized_score_dict['C51'] = np.random.beta(2.5, 3.5, size=(num_runs, num_games)) * 1.2

# REM - ensemble method, good performance
atari_200m_normalized_score_dict['REM'] = np.random.beta(2.7, 3.3, size=(num_runs, num_games)) * 1.3

# Rainbow - combination of improvements, very good
atari_200m_normalized_score_dict['Rainbow'] = np.random.beta(3.0, 3.0, size=(num_runs, num_games)) * 1.4

# IQN - implicit quantile networks, excellent
atari_200m_normalized_score_dict['IQN'] = np.random.beta(3.3, 2.7, size=(num_runs, num_games)) * 1.5

# M-IQN - modified IQN, even better
atari_200m_normalized_score_dict['M-IQN'] = np.random.beta(3.5, 2.5, size=(num_runs, num_games)) * 1.6

# DreamerV2 - model-based method, state of the art
atari_200m_normalized_score_dict['DreamerV2'] = np.random.beta(4, 2, size=(num_runs, num_games)) * 1.7



# Define the aggregate function with the exact metrics shown
aggregate_func = lambda x: np.array([
    metrics.aggregate_median(x),
    metrics.aggregate_iqm(x),
    metrics.aggregate_mean(x),
    metrics.aggregate_optimality_gap(x)])

# Compute aggregate scores and their confidence intervals
aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
    atari_200m_normalized_score_dict, aggregate_func, reps=50000)

# Create figure with dark background to match the image
plt.style.use('dark_background')
fig, axes = plt.subplots(figsize=(12, 8))

# Create the plot
axes = plot_utils.plot_interval_estimates(
    aggregate_scores, aggregate_score_cis,
    metric_names=['Median', 'IQM', 'Mean', 'Optimality Gap'],
    algorithms=algorithms, xlabel='Human Normalized Score',
    ax=axes)

# Set a title for the plot to match the image
plt.suptitle('Aggregate metrics with 95% Stratified Bootstrap CIs', fontsize=16, color='white')
plt.title('IQM, Optimality Gap, Median, Mean', fontsize=14, color='white')

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.85)  # Make room for the titles

# Save the figure
plt.savefig('aggregate_metrics.png', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()

# Print the point estimates for each algorithm and metric
print("Point estimates:")
for i, algo in enumerate(algorithms):
    print(f"{algo}:")
    for j, metric in enumerate(['Median', 'IQM', 'Mean', 'Optimality Gap']):
        print(f"  {metric}: {aggregate_scores[i, j]:.3f} ({aggregate_score_cis[i, j, 0]:.3f}, {aggregate_score_cis[i, j, 1]:.3f})")