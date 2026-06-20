"""
Extract per-residue features (dpocket pocket statistics, MACCS fingerprints,
ESM embeddings) for each CV fold and save them as .npy / .json files.

Run after prepare_reports.py has produced allosteric_ligands_{mode}_fold_{fold}.json.

Outputs written to --output-dir (default=FEATURES_DIR):
    {mode}_{allo|non_allo}_{dpocket|maccs}_fold_{fold}.npy
    {mode}_{allo|non_allo}_res_fold_{fold}.json
    {mode}_{allo|non_allo}_esm_fold_{fold}.npy
"""
import argparse
import json
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from train_scripts.config import (
    FEATURES_DIR, DPOCKET_OUT_DIR, ESM_EMB_DIR,
    MACCS_FPS_PATH, N_FOLDS, DPOCKET_FEATURES,
)

maccs_fps = pickle.load(open(MACCS_FPS_PATH, "rb"))


def get_dpocket_feats(pockets):
    dfs, valid = [], []
    for pocket in pockets:
        path = f"{DPOCKET_OUT_DIR}/{pocket}/{pocket}.txt"
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, sep=r"\s+")[DPOCKET_FEATURES]
            if len(df) > 0:
                dfs.append(df)
                valid.append(pocket)
        except Exception as e:
            print(f"Problem with pocket {pocket}: {e}")
    if not dfs:
        return None
    result = pd.concat(dfs)
    result["pocket"] = valid
    return result


def pocket_stats(all_pockets_df, selected_pockets):
    sub = all_pockets_df[all_pockets_df["pocket"].isin(selected_pockets)]
    numeric = sub.drop(columns=["pocket"], errors="ignore")
    return numeric.describe().drop(index="count").to_numpy().ravel()


def _process_entry(results_data, unp):
    y = results_data[unp]
    allo_feats, allo_maccs, allo_res = [], [], []
    non_allo_feats, non_allo_maccs, non_allo_res = [], [], []

    for group, feats_list, maccs_list, res_list in [
        ("Allosteric", allo_feats, allo_maccs, allo_res),
        ("Non-Allosteric", non_allo_feats, non_allo_maccs, non_allo_res),
    ]:
        try:
            pocket_df = get_dpocket_feats(y[f"{group} ligands"])
        except Exception:
            print(f"Problem getting dpocket feats for {group} pockets of {unp}")
            pocket_df = None

        if pocket_df is None:
            continue

        for residue in y[f"{group} residues"]:
            for res_name, selected_pockets in residue.items():
                if not selected_pockets:
                    continue
                try:
                    feats_list.append(pocket_stats(pocket_df, selected_pockets).reshape(1, -1))
                    fp = np.concatenate(
                        [maccs_fps[p.split("-")[2]].reshape(1, -1) for p in selected_pockets],
                        axis=0,
                    ).max(axis=0).reshape(1, -1)
                    maccs_list.append(fp)
                    res_list.append(f"{unp}-{res_name}")
                except Exception:
                    print(f"Problem at residue {res_name} in {unp}")

    return allo_feats, allo_maccs, allo_res, non_allo_feats, non_allo_maccs, non_allo_res


def extract_dpocket_maccs(mode, fold, output_dir):
    results_data = json.load(
        open(f"{FEATURES_DIR}/allosteric_ligands_{mode}_fold_{fold}.json")
    )
    raw = Parallel(n_jobs=-1, backend="multiprocessing", verbose=1)(
        delayed(_process_entry)(results_data, unp) for unp in results_data
    )

    allo_feats, allo_maccs, allo_res = [], [], []
    non_allo_feats, non_allo_maccs, non_allo_res = [], [], []
    for a_f, a_m, a_r, na_f, na_m, na_r in raw:
        allo_feats.extend(a_f); allo_maccs.extend(a_m); allo_res.extend(a_r)
        non_allo_feats.extend(na_f); non_allo_maccs.extend(na_m); non_allo_res.extend(na_r)

    np.save(f"{output_dir}/{mode}_allo_dpocket_fold_{fold}.npy", np.concatenate(allo_feats, axis=0))
    np.save(f"{output_dir}/{mode}_non_allo_dpocket_fold_{fold}.npy", np.concatenate(non_allo_feats, axis=0))
    np.save(f"{output_dir}/{mode}_allo_maccs_fold_{fold}.npy", np.concatenate(allo_maccs, axis=0))
    np.save(f"{output_dir}/{mode}_non_allo_maccs_fold_{fold}.npy", np.concatenate(non_allo_maccs, axis=0))
    json.dump(allo_res, open(f"{output_dir}/{mode}_allo_res_fold_{fold}.json", "w"), indent=1)
    json.dump(non_allo_res, open(f"{output_dir}/{mode}_non_allo_res_fold_{fold}.json", "w"), indent=1)
    print(f"[{mode} fold {fold}] allo={len(allo_res)}, non_allo={len(non_allo_res)}")


def load_esm_embs(unp):
    path = f"{ESM_EMB_DIR}/ASD_{unp}_esm_embs.npy"
    return np.load(path) if os.path.exists(path) else None


def extract_esm(mode, fold, output_dir):
    allo_res = json.load(open(f"{output_dir}/{mode}_allo_res_fold_{fold}.json"))
    non_allo_res = json.load(open(f"{output_dir}/{mode}_non_allo_res_fold_{fold}.json"))

    all_unps = list(set(r.split("-")[0] for r in allo_res + non_allo_res))
    esm_cache = {
        unp: load_esm_embs(unp)
        for unp in tqdm(all_unps, desc=f"Loading ESM [{mode} fold {fold}]")
    }

    def gather(residues):
        embs = []
        for r in residues:
            unp, res_name = r.split("-")
            idx = int(res_name[1:]) - 1
            emb = esm_cache.get(unp)
            if emb is not None:
                embs.append(emb[idx, :].reshape(1, -1))
        return np.concatenate(embs, axis=0)

    np.save(f"{output_dir}/{mode}_allo_esm_fold_{fold}.npy", gather(allo_res))
    np.save(f"{output_dir}/{mode}_non_allo_esm_fold_{fold}.npy", gather(non_allo_res))
    print(f"[{mode} fold {fold}] ESM done")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=FEATURES_DIR)
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(N_FOLDS)))
    parser.add_argument("--modes", nargs="+", default=["train", "val"])
    parser.add_argument("--skip-dpocket-maccs", action="store_true")
    parser.add_argument("--skip-esm", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for fold in args.folds:
        for mode in args.modes:
            print(f"\n=== {mode.upper()} fold {fold} ===")
            if not args.skip_dpocket_maccs:
                extract_dpocket_maccs(mode, fold, args.output_dir)
            if not args.skip_esm:
                extract_esm(mode, fold, args.output_dir)


if __name__ == "__main__":
    main()
