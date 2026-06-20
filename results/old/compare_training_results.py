import json
import matplotlib.pyplot as plt

# Define the architectures to compare
architectures = ["geom", "seq", "lig", "geom+seq", "geom+lig", "seq+lig", "seq+geom+lig", "plm","lig_only"]


def analyze_results(metric="auc"):
    # Define the metric to compare — change this to 'accuracy', 'auc', etc.

    # Store results for plotting
    plot_data = {}

    # Load and parse data
    for archi in architectures:
        try:
            with open(f"ASD_gnn_train_results_{archi}.json", "r") as f:
                gnn_results = json.load(f)
        except FileNotFoundError:
            print(f"File for {archi} not found, skipping.")
            continue

        # Assuming you have folds like FOLD_0, FOLD_1, etc.
        metric_values = []
        for fold_key in gnn_results:
            fold_data = gnn_results[fold_key]["val_results"]
            fold_metric_values = [step[metric] for step in fold_data]
            metric_values.append(fold_metric_values)

        # Average across folds (transpose first)
        avg_metric = [sum(x)/len(x) for x in zip(*metric_values)]
        plot_data[archi] = avg_metric

    # Plotting
    plt.figure(figsize=(12, 6))
    for archi, values in plot_data.items():
        plt.plot(range(1, len(values) + 1), values, label=archi)

    plt.title(f"Validation {metric.upper()} over Epochs")
    plt.xlabel("Training Step")
    plt.ylabel(f"Val {metric.upper()}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"ASD_training_{metric}.png")

with open(f"ASD_gnn_train_results_geom.json", "r") as f:
    results = json.load(f)["FOLD_0"]["val_results"][0]

for metric in results.keys():
    print("Analyzing for metric",metric.upper())
    analyze_results(metric)