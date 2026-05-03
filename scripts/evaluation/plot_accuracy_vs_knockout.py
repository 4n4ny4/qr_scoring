
import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter

def load_results(file_path):
    """Loads results from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def get_pooled_accuracy(results):
    """Extracts pooled accuracy from the results structure."""
    knockout_sizes = []
    accuracies = []
    
    # Handle structure of QRScore-SEC_results.json
    if 'pooled_results' in results:
        for k, data in results['pooled_results'].items():
            knockout_sizes.append(int(k))
            accuracies.append(data['accuracy'])
    # Handle structure from comparison_summary.json
    elif 'results' in results: 
        for k, data in results['results'].items():
            knockout_sizes.append(int(k))
            accuracies.append(data['accuracy'])
    # Handle structure from Random-seed*.json
    elif 'accuracy_curve' in results:
        for k, acc in results['accuracy_curve'].items():
            knockout_sizes.append(int(k))
            accuracies.append(acc)

    if not knockout_sizes:
        return [], []

    sorted_data = sorted(zip(knockout_sizes, accuracies))
    return zip(*sorted_data)


def get_random_avg_accuracy(results_dir):
    """Calculates the average accuracy for random baselines."""
    random_files = [f for f in os.listdir(results_dir) if f.startswith('Random-seed') and f.endswith('.json')]
    if not random_files:
        return [], []

    all_accuracies = {}
    for file_name in random_files:
        results = load_results(os.path.join(results_dir, file_name))
        # The random results files have a different structure
        if 'accuracy_curve' in results:
            for k, acc in results['accuracy_curve'].items():
                k = int(k)
                if k not in all_accuracies:
                    all_accuracies[k] = []
                all_accuracies[k].append(acc)

    knockout_sizes = sorted(all_accuracies.keys())
    avg_accuracies = [np.mean(all_accuracies[k]) for k in knockout_sizes]
    
    return knockout_sizes, avg_accuracies


def main():
    parser = argparse.ArgumentParser(description="Plot Head Ablation Accuracy vs. Knockout Size.")
    parser.add_argument('--results_dir', type=str, required=True, help='Directory containing the result JSON files.')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model for the plot title.')
    args = parser.parse_args()

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 6))

    plot_configs = {
        'QRScore-SEC': {'label': 'QRScore-SEC', 'color': '#0077CC', 'marker': 'o'},
        'QRScore-8B-LME-TRAIN': {'label': 'QRScore-8B-LME-TRAIN', 'color': '#5cb85c', 'marker': 's'},
        'QRScore-8B-NQ-TRAIN': {'label': 'QRScore-8B-NQ-TRAIN', 'color': '#9e9d24', 'marker': 'd', 'linestyle': '--'},
    }

    # Load per-method result files (produced by run_ablation.py).
    for method, config in plot_configs.items():
        result_path = os.path.join(args.results_dir, f"{method}_results.json")
        if not os.path.exists(result_path):
            continue
        method_results = load_results(result_path)
        k_values, acc_values = get_pooled_accuracy(method_results)
        if not k_values:
            continue
        ax.plot(
            k_values,
            acc_values,
            label=config['label'],
            color=config['color'],
            marker=config['marker'],
            linestyle=config.get('linestyle', '-'),
        )


    # Plot Random-avg results
    k_random, acc_random = get_random_avg_accuracy(args.results_dir)
    if k_random:
        ax.plot(k_random, acc_random, label='Random-avg', color='gray', marker='x', linestyle=':')

    ax.set_title(f"Head Ablation: Accuracy vs Knockout Size ({args.model_name})\n(Steeper drop = more effective detection method)", fontsize=14)
    ax.set_xlabel("Number of Knocked-Out Heads (K)", fontsize=12)
    ax.set_ylabel("Answer Accuracy", fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.tight_layout()

    output_path = os.path.join(args.results_dir, 'accuracy_vs_knockout.png')
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == '__main__':
    main()
