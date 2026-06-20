"""Cross-validation training with an MLP classifier (PyTorch).

Feature types evaluated: maccs, dpocket, esm, esm+dpocket, esm+dpocket+maccs
"""
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

from ligand_classification.config import FEATURES_DIR, N_FOLDS
from ligand_classification.data.io import load_fold_arrays, clean_dpocket_nans, compose_features
from ligand_classification.models.mlp import MLPClassifier, train_and_evaluate
from ligand_classification.models.metrics import balance_subsample, build_val_arrays

FEAT_TYPES = ["maccs", "dpocket", "esm", "esm+dpocket", "esm+dpocket+maccs"]
COLORS = {
    "maccs": "tab:blue",
    "dpocket": "tab:orange",
    "esm": "tab:green",
    "esm+dpocket": "tab:red",
    "esm+dpocket+maccs": "tab:purple",
}


def run(folds, features_dir, device, num_epochs, hidden_dim, evaluate_each):
    results = defaultdict(lambda: {"train": [], "val": []})

    for fold in folds:
        print(f"\n--- fold {fold} ---")
        arrays = clean_dpocket_nans(load_fold_arrays(fold, features_dir))

        for feat_type in FEAT_TYPES:
            print(f"\n  [{feat_type}]")
            train_allo = compose_features(arrays, feat_type, "train", "allo")
            train_non_allo = compose_features(arrays, feat_type, "train", "non_allo")
            val_allo = compose_features(arrays, feat_type, "val", "allo")
            val_non_allo = compose_features(arrays, feat_type, "val", "non_allo")

            train_feats, y_train = balance_subsample(train_allo, train_non_allo)
            val_feats, y_val = build_val_arrays(val_allo, val_non_allo)

            train_t = torch.tensor(train_feats).float().to(device)
            val_t = torch.tensor(val_feats).float().to(device)

            model = MLPClassifier(train_t.shape[1], hidden_dim)
            tm_list, vm_list = train_and_evaluate(
                model, train_t, y_train, val_t, y_val,
                num_epochs=num_epochs, device=device, evaluate_each=evaluate_each,
            )
            results[feat_type]["train"].append(np.array(tm_list))
            results[feat_type]["val"].append(np.array(vm_list))

    # average across folds
    avg = {}
    for feat_type, data in results.items():
        train_stack = np.stack(data["train"])
        val_stack = np.stack(data["val"])
        avg[feat_type] = {
            "train_mean": train_stack.mean(axis=0),
            "train_std": train_stack.std(axis=0),
            "val_mean": val_stack.mean(axis=0),
            "val_std": val_stack.std(axis=0),
        }

    x = np.arange(len(next(iter(avg.values()))["train_mean"]))

    for title, key in [("Train MCC", "train"), ("Validation MCC", "val")]:
        plt.figure(figsize=(12, 7))
        for feat_type, stats in avg.items():
            plt.plot(x, stats[f"{key}_mean"], color=COLORS.get(feat_type), label=feat_type)
        plt.xlabel(f"Evaluation step (every {evaluate_each} epochs)")
        plt.ylabel(title)
        plt.title(f"Average {title} across {len(folds)} folds")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        fname = f"{key}_mcc_across_folds.png"
        plt.savefig(fname, dpi=300)
        print(f"Saved {fname}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", default=FEATURES_DIR)
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(N_FOLDS)))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-epochs", type=int, default=1000)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--evaluate-each", type=int, default=10)
    args = parser.parse_args()
    run(args.folds, args.features_dir, args.device, args.num_epochs, args.hidden_dim, args.evaluate_each)


if __name__ == "__main__":
    main()
