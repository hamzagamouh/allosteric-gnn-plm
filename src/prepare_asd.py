import json,requests
from tqdm import tqdm
HOME_FOLDER="/storage/praha1/home/hamzagamouh"
asd_entries=json.load(open(f"{HOME_FOLDER}/ahoj-allosteric/src/method_2/ASD_entries.json","r"))


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


## Getting Uniprot sequence
# for unp,info in tqdm(asd_entries.items()):
#     try:
#         ref_sequence=get_uniprot_sequence(unp)
#     except:
#         ref_sequence=""

#     asd_entries[unp]["ref_sequence"]=ref_sequence

# json.dump(asd_entries,open(f"{HOME_FOLDER}/ahoj-allosteric/src/method_2/ASD_entries.json","w"),indent=1)

## Make fasta file for MMSeqs

# with open("ASD_seqs.fasta", "w") as f:
#     for unp, info in asd_entries.items():
#         seq=info["ref_sequence"]
#         if seq:
#             if len(seq)>0:
#                 f.write(f">{unp}\n")
#                 f.write(seq + "\n")

## Collect clusters from mmseqs output

# asd_clusters={}
# with open(f"{HOME_FOLDER}/allosteric/method_2/ASD_30_clusterRes_cluster.tsv",'r') as f:
#     for line in f:
#         cluster_id,unp=line.strip().split("\t")
#         if cluster_id in asd_clusters.keys():
#             asd_clusters[cluster_id]+=[unp]
#         else:
#             asd_clusters[cluster_id]=[unp]

# print(len(asd_clusters),"clusters")
# json.dump(asd_clusters,open(f"{HOME_FOLDER}/ahoj-allosteric/src/method_2/ASD_clusters_30.json","w"),indent=1)

## Prepare dataset folds

# asd_clusters=json.load(open(f"{HOME_FOLDER}/ahoj-allosteric/src/method_2/ASD_clusters_30.json","r"))

# cluster_ids=[x for x in asd_clusters.keys()]
# from sklearn.model_selection import train_test_split, KFold

# def make_splits(data, test_size=0.2, n_splits=5, shuffle=True):
#     """
#     Splits data into an independent test set and K-fold CV splits.

#     Parameters:
#     - data (list): Input data.
#     - test_size (float): Proportion of data to use as independent test set.
#     - n_splits (int): Number of CV folds.
#     - random_state (int): Random seed.
#     - shuffle (bool): Whether to shuffle before splitting.

#     Returns:
#     - test_data (list): Independent test set.
#     - cv_splits (list of tuples): Each tuple is (train_indices, val_indices) for one fold.
#     """
#     # Create independent test split
#     train_data, test_data = train_test_split(
#         data, test_size=test_size, shuffle=shuffle
#     )

#     # Generate CV splits on the training data
#     kf = KFold(n_splits=n_splits, shuffle=shuffle)
#     cv_splits = []
#     train_data_indices = list(range(len(train_data)))

#     for train_idx, val_idx in kf.split(train_data_indices):
#         cv_splits.append((
#             [train_data[i] for i in train_idx],
#             [train_data[i] for i in val_idx]
#         ))

#     return test_data, cv_splits

# test,cv_splits=make_splits(cluster_ids)

# import os
# test_data=[]
# for cluster_id in test:
#     for x in asd_clusters[cluster_id]:
#         if os.path.exists(f"{HOME_FOLDER}/allosteric/method_2/ASD_{x}_esm_embs.npy"):
#             test_data+=[x]

# cross_val=[]
# for split in cv_splits:
#     train,val=split
#     d={"train":[],"val":[]}
#     for x in train:
#         if os.path.exists(f"{HOME_FOLDER}/allosteric/method_2/ASD_{x}_esm_embs.npy"):
#             d["train"]+=[x]
#     for x in val:
#         if os.path.exists(f"{HOME_FOLDER}/allosteric/method_2/ASD_{x}_esm_embs.npy"):
#             d["val"]+=[x]
#     cross_val+=[d]

# json.dump({"CV":cross_val,"Test":test_data},open(f"{HOME_FOLDER}/allosteric/method_2/ASD_dataset.json","w"),indent=3)



## Prepare ASD dataset labels
import numpy as np
def make_labels(atoms,ligand,th=10):
    if ligand is None:
        return ""
    lig_coords=np.array([x.coord for x in ligand.get_atoms()])
    lig_centroid=np.sum(lig_coords,axis=0)/lig_coords.shape[0]
    labels=[0]*len(atoms)
    idx=[]
    coords=[]
    for k,x in enumerate(atoms):
        if x is not None:
            idx+=[k]
            coords+=[x.coord]
    coords=np.array(coords)
    distances = np.linalg.norm(coords - lig_centroid, axis=1)
    # Assign labels based on threshold
    selected_labels = (distances <= th).astype(int)
    for k,label in enumerate(selected_labels):
        labels[idx[k]]=label
    return "".join([str(x) for x in labels])

import pickle,os

def extract_label(unp,info):
    if os.path.exists(f"{HOME_FOLDER}/allosteric/asd_processing/{unp}_residues.pkl"):
        # try:
        ref_pdb_id,ref_chain_id,mod,mod_id=info["allosteric_modulator"].split("-")
        ref_sequence=info['ref_sequence']
        ref_entry=f"{ref_pdb_id}-{ref_chain_id}"
        residues=pickle.load(open(f"{HOME_FOLDER}/allosteric/asd_processing/{unp}_ref_residues.pkl",'rb'))
        modulator=residues["allosteric_modulator"]
        ca_atoms=residues[ref_entry]
        labels=make_labels(ca_atoms,modulator,th=10)
        # asd_entries[unp]["labels"]=labels
        return labels
            

        # except Exception as e:
        #     asd_entries[unp]["labels"]=[]
        #     print("Error",e)

# from joblib import Parallel,delayed
# all_labels=Parallel(n_jobs=-1, backend='multiprocessing', verbose=1)(
#     delayed(extract_label)(unp,info) for unp,info in asd_entries.items()
# )

# count=0
# for k,unp in enumerate(asd_entries.keys()):
#     if asd_entries[unp]["labels"]!=all_labels[k]:
#         count+=1
#     asd_entries[unp]["labels"]=all_labels[k]

# print('Mismatches',count)

# # print([x["labels"] for x in asd_entries.values()])
# json.dump(asd_entries,open(f"{HOME_FOLDER}/ahoj-allosteric/src/method_2/ASD_entries_labels.json","w"),indent=3)


## Prepare ASD dataset ligands


from Bio.PDB import PDBList, PDBParser, is_aa, Polypeptide
from helper import *
import os
HOME_FOLDER="/storage/praha1/home/hamzagamouh"
PDB_FOLDER=f"{HOME_FOLDER}/allosteric/pdb_files"

parser = PDBParser(QUIET=True)



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


def is_ligand(res):
    return res.id[0] != " " and res.resname != "HOH"
def get_ligands_from_pdb(pdb_id: str):
    filename=f"{PDB_FOLDER}/{pdb_id.lower()}.pdb"
    # Parse structure
    structure = parser.get_structure(pdb_id, filename)

    # Get the model (usually only one)
    model = structure[0]
    
    ligands=[]
    for chain in model:    
        ligands+= [(res,f"{res.get_resname()}-{res.id[1]}") for res in chain if is_ligand(res)] # , standard=True (do it if you want)
    return ligands

from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.rdmolfiles import MolFromPDBBlock


def download_ligand_sdf(ligand_id, filename=None):
    ligand_id = ligand_id.upper()
    url = f"https://files.rcsb.org/ligands/download/{ligand_id}_ideal.sdf"
    if filename is None:
        filename = f"{PDB_FOLDER}/{ligand_id}.sdf"

    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        # print(f"Downloaded: {filename}")
    else:
        print(f"Failed to download {ligand_id}: HTTP {response.status_code}")


def get_lig_mol(lig_name):
    sdf_file=f"{PDB_FOLDER}/{lig_name}.sdf"
    if not os.path.exists(sdf_file):
        download_ligand_sdf(lig_name)
    ligand_mol =[mol for mol in Chem.SDMolSupplier(sdf_file) if mol][0]
    return ligand_mol


def compute_maccs_from_lig_name(lig_name):
    try:
        mol=get_lig_mol(lig_name)
        if mol is None:
            raise ValueError("Could not convert residue to RDKit Mol.")
        fp = MACCSkeys.GenMACCSKeys(mol)
        return np.array(fp)
    except:
        return None


asd_entries=json.load(open(f"{HOME_FOLDER}/ahoj-allosteric/src/method_2/ASD_entries.json","r"))

def assign_lig_feats(res_list,lig_list):
    ca_atoms=[get_ca_atom(res) for res in res_list]
    labels={}
    for ligand in lig_list:
        if ligand[0] is None:
            continue
        labels[ligand[1]]=make_labels(ca_atoms,ligand[0])
    return labels    
def extract_ligands(unp):
    if os.path.exists(f"{HOME_FOLDER}/allosteric/asd_processing/ligand_info/{unp}_ligands.json"):
        return
    try:
        info=asd_entries[unp]
        modulator=info["allosteric_modulator"]
        ref_pdb_id,ref_chain_id,mod,mod_id=modulator.split("-")
        _,sequence,ligand=get_chain_sequence_from_pdb(ref_pdb_id,ref_chain_id,mod,mod_id)
        ref_entry=f"{ref_pdb_id}-{ref_chain_id}"
        info['chains']+=[ref_entry]
        ref_sequence=info["ref_sequence"]
        if ref_sequence is None:
            ref_sequence=sequence
        
        ligand_info={}
        for entry in tqdm(info['chains'][::-1]):
            try:
                ligand_info[entry]={}
                pdb_id,chain_id=entry.split("-")
                residues,sequence,_=get_chain_sequence_from_pdb(pdb_id,chain_id,mod,mod_id)
                assert len(residues)==len(sequence)
                algn=global_alignment(ref_sequence,sequence)
                mapped_residues=map_residues(algn,sequence,residues,ref_sequence)
                pdb_ligands=get_ligands_from_pdb(pdb_id)
                ligand_labels=assign_lig_feats(mapped_residues,pdb_ligands)
                # maccs_fps={lig[1]:compute_maccs_from_lig_name(lig[1]) for lig in pdb_ligands}
                ligand_info[entry]=ligand_labels
            except KeyboardInterrupt:
                import sys
                sys.exit(1)
            except Exception as e:
                print(f"Error in {unp} - {entry}",e)
        
        json.dump(ligand_info,open(f"{HOME_FOLDER}/allosteric/asd_processing/ligand_info/{unp}_ligands.json",'w'),indent=2)
    except Exception as e:
        print(e)
# for unp in tqdm(asd_entries.keys()):
#     extract_ligands(unp)
#     # break

# from joblib import Parallel,delayed
# all_labels=Parallel(n_jobs=-1, backend='multiprocessing', verbose=1)(
#     delayed(extract_ligands)(unp) for unp in asd_entries.keys()
# )


ligand_names=[]
for file in os.listdir(f"{HOME_FOLDER}/allosteric/asd_processing/ligand_info"):
    lig_info=json.load(open(f"{HOME_FOLDER}/allosteric/asd_processing/ligand_info/{file}",'r'))
    for k,v in lig_info.items():
        for lig_name in v.keys():
            ligand_names+=[lig_name.split("-")[0]]

ligand_names=list(set(ligand_names))
print(ligand_names[:10],len(ligand_names))
# maccs_fps={lig_name:compute_maccs_from_lig_name(lig_name) for lig_name in tqdm(ligand_names)}


from joblib import Parallel,delayed
maccs_fps=Parallel(n_jobs=-1, backend='multiprocessing', verbose=1)(
    delayed(compute_maccs_from_lig_name)(lig_name) for lig_name in ligand_names
)

maccs_fps={lig_name:fp for lig_name,fp in zip(ligand_names,maccs_fps)}
pickle.dump(maccs_fps,open(f"{HOME_FOLDER}/allosteric/asd_processing/ligands_maccs.pkl",'wb'))