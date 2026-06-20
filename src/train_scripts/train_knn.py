"""Cross-validation training with a k-NN classifier.

Feature types evaluated: maccs, dpocket, esm
"""
import argparse
import numpy as np
from collections import defaultdict

from train_scripts.config import FEATURES_DIR, N_FOLDS
from train_scripts.data.io import load_fold_arrays, clean_dpocket_nans, compose_features
from train_scripts.models.knn import make_knn
from train_scripts.models.metrics import evaluate_mcc, balance_subsample, build_val_arrays

FEAT_TYPES = ["maccs", "dpocket", "esm"]


def run(folds, features_dir, n_neighbors):
    results = defaultdict(lambda: {"train": [], "val": []})

    for fold in folds:
        print(f"\n--- fold {fold} ---")
        arrays = clean_dpocket_nans(load_fold_arrays(fold, features_dir))

        for feat_type in FEAT_TYPES:
            train_allo = compose_features(arrays, feat_type, "train", "allo")
            train_non_allo = compose_features(arrays, feat_type, "train", "non_allo")
            val_allo = compose_features(arrays, feat_type, "val", "allo")
            val_non_allo = compose_features(arrays, feat_type, "val", "non_allo")

            train_feats, y_train = balance_subsample(train_allo, train_non_allo)
            val_feats, y_val = build_val_arrays(val_allo, val_non_allo)

            knn = make_knn(train_feats, y_train, n_neighbors=n_neighbors)
            tm = evaluate_mcc(knn, train_feats, y_train)
            vm = evaluate_mcc(knn, val_feats, y_val)

            print(f"  {feat_type:20s}  train_mcc={tm:.4f}  val_mcc={vm:.4f}")
            results[feat_type]["train"].append(tm)
            results[feat_type]["val"].append(vm)

    print("\n===== CV SUMMARY =====")
    for feat_type in FEAT_TYPES:
        tm = np.mean(results[feat_type]["train"])
        vm = np.mean(results[feat_type]["val"])
        ts = np.std(results[feat_type]["train"])
        vs = np.std(results[feat_type]["val"])
        print(f"{feat_type:20s}  train={tm:.3f}±{ts:.3f}  val={vm:.3f}±{vs:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", default=FEATURES_DIR)
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(N_FOLDS)))
    parser.add_argument("--n-neighbors", type=int, default=30)
    args = parser.parse_args()
    run(args.folds, args.features_dir, args.n_neighbors)


if __name__ == "__main__":
    main()
