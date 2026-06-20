"""
Generate per-residue allosteric ligand reports for each CV fold.

For each protein entry this script maps every binding-site residue to the
ligands that contact it, then classifies each residue+ligand pair as allosteric
or non-allosteric.

Inputs  (read from FEATURES_DIR):
    {mode}_unps_fold_{fold}.json
    {mode}_y_fold_{fold}.pkl

Outputs (written to FEATURES_DIR):
    allosteric_ligands_{mode}_fold_{fold}.json
"""
import argparse
import json
import pickle
import numpy as np
from tqdm import tqdm

from ligand_classification.config import FEATURES_DIR, N_FOLDS
from ligand_classification.data.asd_data import get_ligands, get_br


def build_report_for_fold(mode, fold, features_dir=FEATURES_DIR):
    unps = json.load(open(f"{features_dir}/{mode}_unps_fold_{fold}.json"))
    y = pickle.load(open(f"{features_dir}/{mode}_y_fold_{fold}.pkl", "rb"))

    results = {}
    for k in tqdm(range(len(unps)), desc=f"{mode} fold {fold}"):
        entry = unps[k]
        labels = y[k]
        for unp, seq in entry.items():
            assert len(seq) == len(labels)
            ligands = get_ligands(unp)
            br = get_br(unp)
            assert len(seq) == len(br)

            allo_idx = np.argwhere((labels == 1) & (br == 1)).flatten()
            non_allo_idx = np.argwhere((labels == 0) & (br == 1)).flatten()

            br_ligands = {i: [] for i in list(allo_idx) + list(non_allo_idx)}
            for chain_id, ligand_info in ligands.items():
                for lig_name, lig_labels in ligand_info.items():
                    lig_labels = np.array([int(x) for x in lig_labels])
                    assert len(lig_labels) == len(seq)
                    for i in list(allo_idx) + list(non_allo_idx):
                        if lig_labels[i] == 1:
                            br_ligands[i].append(f"{chain_id}-{lig_name}")

            final_allo = list(set(
                lig for i in allo_idx for lig in br_ligands[i]
            ))
            final_non_allo = list(
                set(lig for i in non_allo_idx for lig in br_ligands[i]) - set(final_allo)
            )

            results[unp] = {
                "sequence": seq,
                "Allosteric residues": [
                    {f"{seq[i]}{i + 1}": list(set(br_ligands[i]).intersection(final_allo))}
                    for i in allo_idx
                ],
                "Allosteric ligands": final_allo,
                "Non-Allosteric residues": [
                    {f"{seq[i]}{i + 1}": list(set(br_ligands[i]).intersection(final_non_allo))}
                    for i in non_allo_idx
                ],
                "Non-Allosteric ligands": final_non_allo,
            }

    out_path = f"{features_dir}/allosteric_ligands_{mode}_fold_{fold}.json"
    json.dump(results, open(out_path, "w"), indent=4)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", default=FEATURES_DIR)
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(N_FOLDS)))
    parser.add_argument("--modes", nargs="+", default=["train", "val"])
    args = parser.parse_args()

    for mode in args.modes:
        for fold in args.folds:
            build_report_for_fold(mode, fold, args.features_dir)


if __name__ == "__main__":
    main()
