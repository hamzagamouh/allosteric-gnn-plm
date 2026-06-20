import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors


def compute_descriptors(mol):
    if mol is None:
        return None
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "TPSA": Descriptors.TPSA(mol),
        "RotBonds": Descriptors.NumRotatableBonds(mol),
        "Rings": Descriptors.RingCount(mol),
        "HeavyAtoms": Descriptors.HeavyAtomCount(mol),
    }


def compute_descriptor_statistics(mols):
    records = [compute_descriptors(mol) for mol in mols if mol is not None]
    df = pd.DataFrame(records)
    stats = pd.DataFrame({
        "mean": df.mean(),
        "std": df.std(),
        "min": df.min(),
        "max": df.max(),
    })
    return stats, df


def compare_descriptor_kde(
    mols_a,
    mols_b,
    label_a="Allosteric ligands",
    label_b="Non-Allosteric ligands",
    bw_adjust=1.0,
):
    import seaborn as sns

    _, df_a = compute_descriptor_statistics(mols_a)
    _, df_b = compute_descriptor_statistics(mols_b)

    sns.set_style("whitegrid")
    for desc in df_a.columns:
        plt.figure(figsize=(6, 4))
        sns.kdeplot(df_a[desc], label=label_a, bw_adjust=bw_adjust, fill=False, linewidth=2)
        sns.kdeplot(df_b[desc], label=label_b, bw_adjust=bw_adjust, fill=False, linewidth=2)
        plt.xlabel(desc)
        plt.ylabel("Density")
        plt.title(f"KDE of {desc}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"kde_comparison_{desc}.png", dpi=300)
        plt.close()
