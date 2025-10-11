print("Exporting library..")
import sys
sys.path.append("/storage/brno12-cerit/home/hamzagamouh/.local/lib/python3.8/site-packages")


import requests
import pandas as pd
from tqdm import tqdm
import json
from helper import *


def get_pdb_ids_from_uniprot(uniprot_id):
    try:
        url=f"https://www.ebi.ac.uk/pdbe/graph-api/uniprot/unipdb/{uniprot_id}"

        response = requests.get(url)
        response.raise_for_status()

        result = response.json()
        # print(result)
        pdb_ids = [item["name"].lower() for item in result[uniprot_id]["data"]]
        return pdb_ids
    except KeyboardInterrupt:
        import sys
        sys.exit(1)
    except Exception as e:
        print(e)
        return []
    

def get_pdb_uniprot_mapping(pdb_id, uniprot_id):
    try:
        url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id.lower()}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        mappings = data.get(pdb_id.lower(), {}).get("UniProt", {})
        if uniprot_id in mappings.keys():
            residues = mappings[uniprot_id]["mappings"]
        else:
            residues=[]
        return residues
    except Exception as e:
        print(f"Error for {uniprot_id}-{pdb_id}")
        print(f"{e}")
        return []


def get_pdb_residue_from_uniprot(residues,uniprot_residue_number, res_name):
    all_res=[]
    for res in residues:
        if res["unp_start"] <= uniprot_residue_number <= res["unp_end"]:
            pdb_res_num = res["start"]["residue_number"] + (uniprot_residue_number - res["unp_start"])
            pdb_res_id = f"{res_name}-{res['chain_id']}-{pdb_res_num}"
            all_res+=[pdb_res_id]
    return ",".join(all_res)




from Bio.PDB import PDBParser
from Bio.PDB.PDBParser import PDBConstructionWarning
import warnings
import re

# Suppress warnings
warnings.simplefilter('ignore', PDBConstructionWarning)
HOME_FOLDER="/storage/praha1/home/hamzagamouh"
PDB_FOLDER=f"{HOME_FOLDER}/allosteric/pdb_files"

from Bio.PDB import *

PDB_PARSER = PDBParser()

# Create an instance of PDBList
pdb_list = PDBList(server="https://files.rcsb.org",verbose=False)


def download_pdb_structure(pdb_id,suff=""):
    # Fetch the PDB file by its ID (e.g., "1XYZ")
    try:
        if not os.path.exists(f"{PDB_FOLDER}/{pdb_id}{suff}.pdb"):
            pdb_file = pdb_list.retrieve_pdb_file(pdb_id.upper(), pdir=PDB_FOLDER, file_format="pdb")
            pdb_filename=os.path.basename(pdb_file)
            os.rename(pdb_file,f"{pdb_file}".replace(pdb_filename,f"{pdb_id}{suff}.pdb"))
            # print(f"PDB file {pdb_file} has been downloaded.")
    except Exception as e:
        pass 

def extract_chains_match_unp(pdb_id,unp):
    # Path to your PDB file
    pdb_file = f"{PDB_FOLDER}/{pdb_id}.pdb"
    # Read the PDB file as text to access the header
    with open(pdb_file, "r") as file:
        pdb_lines = file.readlines()

    # Dictionary to store chain: uniprot mapping
    chain_uniprot = {}

    # Parse DBREF lines
    for line in pdb_lines:
        if line.startswith("DBREF"):
            # Ensure it's a UniProt reference
            db_name = line[26:32].strip()
            if db_name == "UNP":
                chain_id = line[12].strip()
                uniprot_id = line[33:41].strip()
                chain_uniprot[chain_id] = uniprot_id

    # Display results
    chains=[]
    for chain, uniprot in chain_uniprot.items():
        if unp==uniprot:
            chains+=[f"{pdb_id}-{chain}"]
    return chains


from Bio.PDB import PDBList, PDBParser, is_aa, Polypeptide
import os

parser = PDBParser(QUIET=True)



def get_chain_sequence_from_pdb(pdb_id: str, chain_id: str, resname: str, resnum: int):
    """
    Download a PDB file, parse the structure, and extract:
    - Residue objects for a given chain
    - One-letter amino acid sequence for that chain

    Args:
        pdb_id (str): The 4-letter PDB ID (e.g., "1A8M")
        chain_id (str): The chain identifier (e.g., "A")

    Returns:
        tuple: (list of residue objects, one-letter sequence as string)
    """
    filename=f"{PDB_FOLDER}/{pdb_id.lower()}.pdb"
    # Parse structure
    structure = parser.get_structure(pdb_id, filename)

    # Get the model (usually only one)
    model = structure[0]
    
    # Get the chain
    chain = model[chain_id]
    
    residues = [res for res in chain if is_aa(res)] # , standard=True (do it if you want)

    # Convert to one-letter code
    try:
        sequence = Polypeptide.Polypeptide(residues).get_sequence()
    except Exception:
        sequence = "".join([Polypeptide.three_to_one(res.get_resname()) for res in residues])

    # ligands = []
    ligand=None

    if resname is not None:
        for res in chain:
            hetfield, resseq, icode = res.get_id()
            if str(resseq) == str(resnum) and res.get_resname().strip() == resname.upper():
                # Ensure it's a ligand (i.e., hetfield is not ' ' and not water)
                if hetfield != " " and res.get_resname() != "HOH":
                    ligand=res

    return residues, str(sequence), ligand



#### Collect and download all PDBs

# asd=pd.read_csv("ASD_queries.csv",sep="\t")

# all_pdbs=[]
# for unp in tqdm(asd["UNP"].unique()):
#     all_pdbs+=get_pdb_ids_from_uniprot(unp.strip())
    
# json.dump(list(set(all_pdbs)),open("ASD_pdbs.json","w"),indent=1)
# all_pdbs=json.load(open(f"{HOME_FOLDER}/ahoj-allosteric/src/method_2/ASD_pdbs.json","r"))

# for pdb_id in tqdm(all_pdbs):
#     download_pdb_structure(pdb_id)


#### Extract PDB chains matching the Uniprot IDs 
# asd=pd.read_csv("ASD_queries.csv",sep="\t")

# all_chains={}
# for unp in tqdm(asd["UNP"].unique()):
#     all_chains[unp]=[]
#     pdbs=get_pdb_ids_from_uniprot(unp.strip())
#     if len(pdbs)==0:
#         continue
#     for pdb_id in pdbs:
#         try:
#             all_chains[unp]+=extract_chains_match_unp(pdb_id,unp)
#         except:
#             continue
# json.dump(all_chains,open(f"{HOME_FOLDER}/ahoj-allosteric/src/method_2/ASD_chains.json","w"),indent=1)


#### Make ASD entries

# asd_chains=json.load(open(f"{HOME_FOLDER}/ahoj-allosteric/src/method_2/ASD_chains.json","r"))
# asd=pd.read_csv("ASD_queries.csv",sep="\t")

# asd_entries={}
# for i in range(asd.shape[0]):
#     entry=asd.iloc[i,:].to_dict()
#     unp=entry["UNP"]
#     mod=entry["MOD"]
#     mod_id=entry["MOD_ID"]
#     chain=entry["CHAIN_ID"]
#     pdb_id=entry["PDB_ID"]
#     asd_entries[unp]={"allosteric_modulator":f"{pdb_id}-{chain}-{mod}-{mod_id}",
#                       "chains":asd_chains[unp]}

# json.dump(asd_entries,open(f"{HOME_FOLDER}/ahoj-allosteric/src/method_2/ASD_entries.json","w"),indent=1)

def get_uniprot_sequence(uniprot_id):
    try:
        url=f"https://www.ebi.ac.uk/pdbe/graph-api/uniprot/unipdb/{uniprot_id}"

        response = requests.get(url)
        response.raise_for_status()

        result = response.json()
        return result[uniprot_id]["sequence"]
    except KeyboardInterrupt:
        import sys
        sys.exit(1)
    except Exception as e:
        print(e)
        return None

## Sequence + Coordinates extraction and mapping
from Bio import Align
ALIGNER = Align.PairwiseAligner()

def residue_name(res):
    if res:
        try:
            return three_to_one(res.get_resname())
        except:
            return "X"
    else:
        ""


def global_alignment(seq1, seq2):
    alignments=ALIGNER.align(seq1,seq2)
    return alignments[0]



def map_residues(alignment,sequence,residues,ref_seq):
    aln_ref_idx,aln_seq_idx=alignment.aligned
    final_residues=[None]*len(ref_seq)
    
    for (a,b),(c,d) in zip(aln_ref_idx,aln_seq_idx):
        for k_ref,k_seq in zip(range(a,b),range(c,d)):
            final_residues[k_ref]=residues[k_seq]

    return final_residues

import copy

def get_ca_atom(residue):
    """
    Extracts the CA (alpha carbon) atom from a given residue.

    Args:
        residue (Bio.PDB.Residue): The residue object.

    Returns:
        Bio.PDB.Atom: The CA atom object, or None if CA atom is not found.
    """
    try:
        if residue.has_id('CA'):
            return residue['CA']
        else:
            None
    except:
        return None

from Bio.PDB import Superimposer
from Bio.SVDSuperimposer import SVDSuperimposer

def superimpose_residues(ref_residues,residues):
    assert len(ref_residues)==len(residues)
    final_ca_atoms=[None]*len(residues)
    ref_ca_atoms=[]
    ca_atoms=[]
    superimposed_idx=[]
    k=0
    for x,y in zip(ref_residues,residues):
        atom1=get_ca_atom(x)
        atom2=get_ca_atom(y)
        if atom1 is not None and atom2 is not None:
            ref_ca_atoms+=[atom1]
            ca_atoms+=[atom2]
            superimposed_idx+=[k]
        k+=1
    super_imposer = Superimposer()
    assert len(ref_ca_atoms)==len(ca_atoms)
    super_imposer.set_atoms(ref_ca_atoms,ca_atoms)
    super_imposer.apply(ca_atoms)
    for i,k in enumerate(superimposed_idx):
        final_ca_atoms[k]=ca_atoms[i]
    return final_ca_atoms


def process_entry(unp):
    try:
        info=asd_entries[unp]
        # print(unp)
        modulator=info["allosteric_modulator"]
        ref_pdb_id,ref_chain_id,mod,mod_id=modulator.split("-")
        _,sequence,ligand=get_chain_sequence_from_pdb(ref_pdb_id,ref_chain_id,mod,mod_id)
        ref_entry=f"{ref_pdb_id}-{ref_chain_id}"
        info['chains']+=[ref_entry]
        all_residues={}
        ref_sequence=info["ref_sequence"]
        if ref_sequence is None:
            ref_sequence=sequence
        for entry in tqdm(info['chains'][::-1]):
            # try:
            pdb_id,chain_id=entry.split("-")
            residues,sequence,_=get_chain_sequence_from_pdb(pdb_id,chain_id,mod,mod_id)
            assert len(residues)==len(sequence)
            algn=global_alignment(ref_sequence,sequence)
            mapped_residues=map_residues(algn,sequence,residues,ref_sequence)
            all_residues[entry]=mapped_residues
            # except Exception as e:
            #     print(e)

        ref_residues=all_residues[ref_entry]
        final_residues={"allosteric_modulator":ligand}
        for entry,residues in all_residues.items():
            # print(len(residues),len(ref_residues))
            # try:
            super_ca_atoms=superimpose_residues(ref_residues,residues)
            final_residues[entry]=super_ca_atoms
            # except Exception as e:
                # print(e)

        pickle.dump(final_residues,open(f"{HOME_FOLDER}/allosteric/asd_processing/{unp}_residues.pkl",'wb'))        
    except KeyboardInterrupt:
        sys.exit(0)
    
    except Exception as e:
        print(f"Error in {unp} --> {e}")
        pass


if __name__=="__main__":
    import pickle

    asd_entries=json.load(open(f"{HOME_FOLDER}/ahoj-allosteric/src/method_2/ASD_entries.json","r"))

    from joblib import Parallel,delayed
    
    entries=[unp for unp in asd_entries.keys() if not os.path.exists(f"{HOME_FOLDER}/allosteric/asd_processing/{unp}_residues.pkl")]
    # for unp in tqdm(entries):
    #     process_entry(unp)
    #     break
    Parallel(n_jobs=-1, backend='multiprocessing', verbose=1)(
        delayed(process_entry)(unp) for unp in entries
    )
        
