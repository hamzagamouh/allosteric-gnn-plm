import sys
sys.path.append("/storage/brno12-cerit/home/hamzagamouh/.local/lib/python3.8/site-packages")

import argparse
import os
import shutil
import subprocess

from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Residue import Residue
from Bio.PDB.Chain import Chain

import warnings
warnings.filterwarnings("ignore")
import json

import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import os
HOME_FOLDER="/storage/praha1/home/hamzagamouh"

import dgl
import torch

import json

import pandas as pd
import pickle
from e_gnn_utils import get_atoms_and_features,build_protein_graph,featurize_sequence

import numpy as np
from Bio import PDB
from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB import DSSP
from Bio.PDB.Polypeptide import one_to_three,three_to_one,is_aa
import requests
import os

pocket_mapping=json.load(open(f"{HOME_FOLDER}/allosteric/ahoj_processing/pocket_mapping.json",'r'))



# ahoj_queries=json.load(open(f"{HOME_FOLDER}/allosteric/ahoj_processing/non_catalytic_csa_pockets.json",'r'))
ahoj_queries=json.load(open(f"{HOME_FOLDER}/allosteric/ahoj_processing/non_catalytic_csa_pockets_info.json",'r'))

new_df=pd.read_csv(f"{HOME_FOLDER}/allosteric/ahoj_processing/casbench_processed_csa_pockets.csv")

print(len(new_df[new_df["NON_CATALYTIC_CSA_POCKET"]!="N/A"]["Entry"].unique()),"casbench entries")


non_allo_casbench_pockets=new_df[new_df["LABEL"]=="Non_Allosteric"]["NON_CATALYTIC_CSA_POCKET"].unique().tolist()
allo_casbench_pockets=new_df[new_df["LABEL"]=="Allosteric"]["NON_CATALYTIC_CSA_POCKET"].unique().tolist()

casbench_pockets=[x for x in non_allo_casbench_pockets+allo_casbench_pockets if isinstance(x,str)]
def get_csa_entries(casbench_pocket):
    entries=[]
    for entry in pocket_mapping:
        if entry["pocket"]==casbench_pocket:
            entries+=[entry]

    return entries


parser=PDBParser(QUIET=True)
PDB_FOLDER=f"{HOME_FOLDER}/allosteric/pdb_files"
SCRATCHDIR=f"{HOME_FOLDER}/allosteric"

def get_residues_from_pdb(residues_list,pdb_id,suff=""):
    pdb_file=f"{PDB_FOLDER}/{pdb_id}{suff}.pdb"
    if not os.path.exists(pdb_file):
        print(pdb_file,"don't exist")
        return (pdb_id,[])
    structure = parser.get_structure('protein', pdb_file)
    cat_residues={}
    for res_name in residues_list:
        chain_id=res_name.split("-")[1]
        if chain_id in cat_residues.keys():
            cat_residues[chain_id]+=[res_name]
        else:
            cat_residues[chain_id]=[res_name]
    residues=[]
    # Loop through all chains in the structure
    for model in structure:
        for chain in model:
            for chain_id in cat_residues.keys():
                if str(chain.get_id())==chain_id:
                    # Search for residue by name and position
                    for residue in chain:
                        if is_aa(residue):
                            name=residue.get_resname().upper()
                            res_id=residue.get_id()[1]
                            # res_designation=f"{name}-{chain_id}-{res_id}"
                            res_designation=f"XXX-{chain_id}-{res_id}"
                            if res_designation in cat_residues[chain_id]:
                                residues+=[(res_designation,residue)]
    
    return (pdb_id,residues)





def get_cat_res(pdb_id,cat_res):
    residues=pickle.load(open(f"{SCRATCHDIR}/rmsds/all_residues_{pdb_id}.res","rb"))
    assert len(residues)>0
    if len(residues)>0:
        return residues
    else:
        new_cat_res=[]
        for x in cat_res.values():
            new_cat_res+=x
        residues=get_residues_from_pdb(new_cat_res,pdb_id)

def get_allo_res(pdb_id,allo_res):
    pdb_id,residues=get_residues_from_pdb(allo_res,pdb_id)
    return residues


def get_atoms_from_entry(entry):
    pdb_id=entry["structure"]
    all_residues={pdb_id:get_cat_res(pdb_id,entry["pocket_residues"])}
    final_residues={}
    for pdb_id,res in all_residues.items():
        for res_name,res_obj in res:
            final_residues[f"{pdb_id}-{res_name}"]=res_obj
    atoms={}

    for chain_id in entry["pocket_residues"].keys():
        try:
            residues=[]
            pdb_id=entry["structure"]
            residue_names=[]
            for res_name in entry["pocket_residues"][chain_id]:
                if f"{pdb_id}-{res_name}" in final_residues.keys():
                    if f"{pdb_id}-{res_name}" not in residue_names:
                        residue_names+=[f"{pdb_id}-{res_name}"]
                        residues+=[final_residues[f"{pdb_id}-{res_name}"]]
            atoms[chain_id]=residues
        except Exception as e:
            print(e)
            pass
    return atoms


def get_state_label(state):
    if state=="apo":
        return 0
    else:
        return 1


from Bio.PDB import Structure, Model, Chain, PDBIO

def write_residues_to_pdb(residues, output_filename, chain_id='A'):
    """
    Write a list of Bio.PDB.Residue objects to a PDB file.

    Parameters:
    - residues: list of Bio.PDB.Residue objects
    - output_filename: str, path to output PDB file
    - chain_id: str, single character chain identifier (default 'A')
    """
    # Create new structure, model, and chain
    structure = Structure.Structure('new_structure')
    model = Model.Model(0)
    structure.add(model)
    chain = Chain.Chain(chain_id)
    model.add(chain)

    # Add residues to chain
    for residue in residues:
        chain.add(residue)

    # Write to file
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_filename)


def process_catalytic_pocket(casbench_pocket):
    for entry in tqdm(get_csa_entries(casbench_pocket)):
        try:
            entry_atoms=get_atoms_from_entry(entry)
        except Exception as e:
            print(e)
            continue
        state=get_state_label(entry["state"])
        for chain_id,residues in entry_atoms.items():
            pdb_id=entry['structure']
            pdb_file=f"{pdb_id}.pdb"
            make_fake_res(pdb_file,chain_id,residues)
    




def make_fake_res(pdb_file,chain_id,residues):
    # MODIFIED PDB WITH FAKE "LIGAND" RESIDUE TO DEFINE POCKET
    mod_pdb = pdb_file + f'chain_{chain_id}_mod'
    mod_pdb=f"{HOME_FOLDER}/allosteric/method_2/dpocket_feats/{mod_pdb}"
    structure = PDBParser(QUIET=True).get_structure('_', f"{PDB_FOLDER}/{pdb_file}")
    # FAKE RESIDUE FOR DPOCKET POCKET DEFINITION
    fake = Residue(('H_STP', 9999, ' '), 'STP', 1)
    atom_counter = 0
    for residue in residues:
        for atom in residue.get_atoms():

            # MAKE A FAKE ATOM FROM THE REAL ONE,
            # AND ADD THIS TO THE FAKE RESIDUE
            copy = atom.copy()

            copy.id = 'X' + str(atom_counter)
            copy.element = 'H'
            copy.name = 'X'
            copy.serial_number = max(
                [a.serial_number for a in structure.get_atoms()]) + 1

            copy.detach_parent()
            fake.add(copy)
            atom_counter += 1

    # ADD FAKE RESIDUE TO FIRST CHAIN IN STRUCTURE
    structure[0].get_list()[0].add(fake)

    # SAVE MODIFIED PDB
    io = PDBIO()
    io.set_structure(structure)
    io.save(mod_pdb)

    # MAKE THE DPOCKET INPUT FILE
    input_file = mod_pdb + '.input'

    with open(input_file, 'w') as fo:
        fo.write('{}\tSTP'.format(mod_pdb))

print(len(casbench_pockets),"casbench_pockets")
for casbench_pocket in tqdm(casbench_pockets):
    process_catalytic_pocket(casbench_pocket)