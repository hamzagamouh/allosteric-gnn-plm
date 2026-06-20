"""Utilities for loading pre-extracted feature arrays for a given CV fold."""
import json
import numpy as np
from train_scripts.config import FEATURES_DIR, DPOCKET_VALID_COLS, DPOCKET_FEATURES

FEAT_TYPES = ("maccs", "esm", "dpocket")
LABELS = ("allo", "non_allo")
MODES = ("train", "val")

_DPOCKET_STATS = ["mean", "std", "min", "25%", "50%", "75%", "max"]
_ALL_DPOCKET_NAMES = [f"{feat}-{stat}" for feat in DPOCKET_FEATURES for stat in _DPOCKET_STATS]
DPOCKET_FEATURE_NAMES = [_ALL_DPOCKET_NAMES[i] for i in DPOCKET_VALID_COLS]

# MACCS fingerprint legend (166 structural keys, RDKit implementation)
# -----------------------------------------------------------------------
# Atom symbols:
#   A  Any valid periodic table element
#   Q  Heteroatom (non-C, non-H)
#   X  Halogen (F, Cl, Br, I)
#   Z  Other (not H, C, N, O, Si, P, S, F, Cl, Br, I)
#
# Bond types:
#   -   Single          =  Double        T / #  Triple
#   ~   Single or double (query)         %      Aromatic (query)
#   $   Ring bond (e.g. $- is a ring single bond)
#   !   Chain / non-ring bond
#   @N  Ring closure back to atom N in the SMARTS pattern
#
# Aromatic conventions:
#   Kekule : alternating single/double bonds in 6-membered rings
#   Arom5  : two double bonds + heteroatom at apex of 5-membered ring
#
# Bit 0 is a dummy bit (always 0); bits 1-166 encode structural features.
# -----------------------------------------------------------------------
MACCS_KEY_NAMES = [
    "DUMMY", "ISOTOPE", "103 < ATOMIC NO. < 256", "GROUP IVA,VA,VIA PERIODS 4-6 (Ge...)",
    "ACTINIDE", "GROUP IIIB,IVB (Sc...)", "LANTHANIDE", "GROUP VB,VIB,VIIB (V...)",
    "QAAA@1", "GROUP VIII (Fe...)", "GROUP IIA (ALKALINE EARTH)", "4M RING",
    "GROUP IB,IIB (Cu...)", "ON(C)C", "S-S", "OC(O)O", "QAA@1", "CTC",
    "GROUP IIIA (B...)", "7M RING", "SI", "C=C(Q)Q", "3M RING", "NC(O)O", "N-O",
    "NC(N)N", "C$=C($A)$A", "I", "QCH2Q", "P", "CQ(C)(C)A", "QX", "CSN", "NS",
    "CH2=A", "GROUP IA (ALKALI METAL)", "S HETEROCYCLE", "NC(O)N", "NC(C)N",
    "OS(O)O", "S-O", "CTN", "F", "QHAQH", "OTHER", "C=CN", "BR", "SAN", "OQ(O)O",
    "CHARGE", "C=C(C)C", "CSO", "NN", "QHAAAQH", "QHAAQH", "OSO", "ON(O)C",
    "O HETEROCYCLE", "QSQ", "Snot%A%A", "S=O", "AS(A)A", "A$A!A$A", "N=O",
    "A$A!S", "C%N", "CC(C)(C)A", "QS", "QHQH (&...)", "QQH", "QNQ", "NO", "OAAO",
    "S=A", "CH3ACH3", "A!N$A", "C=C(A)A", "NAN", "C=N", "NAAN", "NAAAN", "SA(A)A",
    "ACH2QH", "QAAAA@1", "NH2", "CN(C)C", "CH2QCH2", "X!A$A", "S", "OAAAO",
    "QHAACH2A", "QHAAACH2A", "OC(N)C", "QCH3", "QN", "NAAO", "5M RING", "NAAAO",
    "QAAAAA@1", "C=C", "ACH2N", "8M RING", "QO", "CL", "QHACH2A", "A$A($A)$A",
    "QA(Q)Q", "XA(A)A", "CH3AAACH2A", "ACH2O", "NCO", "NACH2A", "AA(A)(A)A",
    "Onot%A%A", "CH3CH2A", "CH3ACH2A", "CH3AACH2A", "NAO", "ACH2CH2A > 1", "N=A",
    "HETEROCYCLIC ATOM > 1 (&...)", "N HETEROCYCLE", "AN(A)A", "OCO", "QQ",
    "AROMATIC RING > 1", "A!O!A", "A$A!O > 1 (&...)", "ACH2AAACH2A", "ACH2AACH2A",
    "QQ > 1 (&...)", "QH > 1", "OACH2A", "A$A!N", "X (HALOGEN)", "Nnot%A%A",
    "O=A > 1", "HETEROCYCLE", "QCH2A > 1 (&...)", "OH", "O > 3 (&...)",
    "CH3 > 2 (&...)", "N > 1", "A$A!O", "Anot%A%Anot%A", "6M RING > 1", "O > 2",
    "ACH2CH2A", "AQ(A)A", "CH3 > 1", "A!A$A!A", "NH", "OC(C)C", "QCH2A", "C=O",
    "A!CH2!A", "NA(A)A", "C-O", "C-N", "O > 1", "CH3", "N", "AROMATIC",
    "6M RING", "O", "RING", "FRAGMENTS",
]
MACCS_FEATURE_NAMES = [f"MACCS-{i} {name}" for i, name in enumerate(MACCS_KEY_NAMES)]


def get_feature_names(feat_type, arrays):
    """Return human-readable feature names matching the columns of compose_features().

    ESM dim is inferred from the array shape; MACCS uses the standard 167-bit key definitions.
    """
    esm_dim = arrays["train_allo_esm"].shape[1]
    esm_names = [f"ESM-dim{i + 1}" for i in range(esm_dim)]

    if feat_type == "maccs":
        return MACCS_FEATURE_NAMES
    if feat_type == "dpocket":
        return DPOCKET_FEATURE_NAMES
    if feat_type == "esm":
        return esm_names
    if feat_type == "esm+dpocket":
        return esm_names + DPOCKET_FEATURE_NAMES
    if feat_type == "esm+dpocket+maccs":
        return esm_names + DPOCKET_FEATURE_NAMES + MACCS_FEATURE_NAMES
    raise ValueError(f"Unknown feat_type: {feat_type}")


def load_fold_arrays(fold, features_dir=FEATURES_DIR):
    """Load all pre-extracted .npy arrays for one fold into a flat dict.

    Keys follow the pattern: "{mode}_{label}_{feat}"
    e.g. "train_allo_dpocket", "val_non_allo_maccs"
    """
    arrays = {}
    for mode in MODES:
        for label in LABELS:
            for feat in FEAT_TYPES:
                key = f"{mode}_{label}_{feat}"
                arrays[key] = np.load(f"{features_dir}/{key}_fold_{fold}.npy")
    return arrays


def load_residue_lists(fold, features_dir=FEATURES_DIR):
    """Load per-residue identifier lists for one fold."""
    res = {}
    for mode in MODES:
        for label in LABELS:
            key = f"{mode}_{label}_res"
            res[key] = json.load(open(f"{features_dir}/{key}_fold_{fold}.json"))
    return res


def clean_dpocket_nans(arrays, valid_cols=DPOCKET_VALID_COLS):
    """Drop NaN-heavy dpocket columns and remove rows that still contain NaNs.

    Returns a new dict with the same keys but cleaned arrays.
    Each (mode, label) group is cleaned independently so row counts stay consistent
    across all feature types within that group.
    """
    clean = {}
    for mode in MODES:
        for label in LABELS:
            dp_key = f"{mode}_{label}_dpocket"
            dp = arrays[dp_key][:, valid_cols]
            nan_mask = np.isnan(dp).any(axis=1)
            keep = ~nan_mask

            clean[dp_key] = dp[keep]
            for feat in ("maccs", "esm"):
                k = f"{mode}_{label}_{feat}"
                clean[k] = arrays[k][keep]

    return clean


def compose_features(arrays, feat_type, mode, label):
    """Return the feature matrix for a given (feat_type, mode, label) combination."""
    if feat_type == "maccs":
        return arrays[f"{mode}_{label}_maccs"]
    if feat_type == "dpocket":
        return arrays[f"{mode}_{label}_dpocket"]
    if feat_type == "esm":
        return arrays[f"{mode}_{label}_esm"]
    if feat_type == "esm+dpocket":
        return np.concatenate(
            [arrays[f"{mode}_{label}_esm"], arrays[f"{mode}_{label}_dpocket"]], axis=1
        )
    if feat_type == "esm+dpocket+maccs":
        return np.concatenate(
            [arrays[f"{mode}_{label}_esm"],
             arrays[f"{mode}_{label}_dpocket"],
             arrays[f"{mode}_{label}_maccs"]], axis=1
        )
    raise ValueError(f"Unknown feat_type: {feat_type}")
