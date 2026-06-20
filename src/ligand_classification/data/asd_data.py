import json
import numpy as np
from ligand_classification.config import ASD_ENTRIES_PATH, LIGAND_INFO_DIR, PDB_FOLDER

_asd_entries = None


def _load_asd_entries():
    global _asd_entries
    if _asd_entries is None:
        _asd_entries = json.load(open(ASD_ENTRIES_PATH))
    return _asd_entries


def get_ligands(unp):
    return json.load(open(f"{LIGAND_INFO_DIR}/{unp}_ligands.json"))


def get_br(unp):
    entries = _load_asd_entries()
    return np.array([int(x) for x in entries[unp]["binding_residues"]])


def get_lig_mol(lig_name):
    from rdkit import Chem
    sdf_file = f"{PDB_FOLDER}/{lig_name}.sdf"
    return [mol for mol in Chem.SDMolSupplier(sdf_file) if mol][0]
