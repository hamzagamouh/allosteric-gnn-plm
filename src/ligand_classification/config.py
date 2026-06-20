HOME_FOLDER = "/storage/praha1/home/hamzagamouh"

# Raw data paths
PDB_FOLDER = f"{HOME_FOLDER}/allosteric/pdb_files"
LIGAND_INFO_DIR = f"{HOME_FOLDER}/allosteric/asd_processing/ligand_info"
MACCS_FPS_PATH = f"{HOME_FOLDER}/allosteric/asd_processing/ligands_maccs.pkl"
ASD_ENTRIES_PATH = f"{HOME_FOLDER}/ahoj-allosteric/src/method_2/ASD_entries.json"

# dpocket
DPOCKET_OUT_DIR = f"{HOME_FOLDER}/allosteric/method_2/dpocket_feats/outputs"
DPOCKET_INP_DIR = f"{HOME_FOLDER}/allosteric/method_2/dpocket_feats/inputs"
DPOCKET_POCKET_PDBS_DIR = f"{HOME_FOLDER}/allosteric/method_2/dpocket_feats/pocket_pdbs"
DPOCKET_EXEC = f"{HOME_FOLDER}/fpocket_sandbox/usr/local/bin/dpocket"

# ESM embeddings
ESM_EMB_DIR = f"{HOME_FOLDER}/allosteric/method_2"

# Fold JSON files and pre-extracted feature arrays live here
# (allosteric_ligands_{mode}_fold_{fold}.json, {mode}_{label}_{feat}_fold_{fold}.npy, etc.)
FEATURES_DIR = f"{HOME_FOLDER}/ahoj-allosteric/src/method_2/ligand_analysis"

N_FOLDS = 5

DPOCKET_FEATURES = (
    "pock_vol nb_AS nb_AS_norm mean_as_ray mean_as_solv_acc apol_as_prop "
    "apol_as_prop_norm mean_loc_hyd_dens mean_loc_hyd_dens_norm hydrophobicity_score "
    "volume_score polarity_score polarity_score_norm charge_score flex prop_polar_atm "
    "as_density as_density_norm as_max_dst as_max_dst_norm drug_score convex_hull_volume "
    "surf_pol_vdw14 surf_pol_vdw22 surf_apol_vdw14 surf_apol_vdw22 n_abpa"
).split()

# Columns 27-53 (second stat-block) are systematically NaN-heavy; drop them by default.
# Each of 27 dpocket features has 7 stats → 189 total columns.
# valid_cols keeps [0..26] + [54..188]
DPOCKET_VALID_COLS = list(range(27)) + list(range(54, len(DPOCKET_FEATURES) * 7))
