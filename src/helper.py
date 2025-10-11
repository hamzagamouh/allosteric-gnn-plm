import numpy as np
from Bio import PDB
from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB import DSSP
from Bio.PDB.Polypeptide import one_to_three,three_to_one,is_aa
import requests
import os

parser=PDBParser()

HOME_FOLDER="/storage/praha1/home/hamzagamouh"

PDB_FOLDER=f"{HOME_FOLDER}/allosteric/pdb_files"
# SCRATCHDIR=os.getenv("SCRATCHDIR",f"{HOME_FOLDER}/allosteric")
# PDB_FOLDER=f"{SCRATCHDIR}/pdb_files"

print("PDB_FOLDER",PDB_FOLDER)



def get_residues(pdb_id,suff=""):
    pdb_file=f"{PDB_FOLDER}/{pdb_id}{suff}.pdb"
    structure = parser.get_structure('protein', pdb_file)
    residues={}
    # Loop through all chains in the structure
    for model in structure:
        for chain in model:
            chain_id=chain.get_id()
            # Search for residue by name and position
            for residue in chain:
                if is_aa(residue):
                    name=residue.get_resname().upper()
                    res_id=residue.get_id()[1]
                    residues[f"{name}-{chain_id}-{res_id}"]=residue
    
    return residues


def get_catalytic_residues(catalytic_residues,pdb_id,chain_id,suff=""):
    pdb_file=f"{PDB_FOLDER}/{pdb_id}{suff}.pdb"
    structure = parser.get_structure('protein', pdb_file)
    residues=[]
    # Loop through all chains in the structure
    for model in structure:
        for chain in model:
            if str(chain.get_id())==chain_id:
                # Search for residue by name and position
                for residue in chain:
                    if is_aa(residue):
                        name=residue.get_resname().upper()
                        res_id=residue.get_id()[1]
                        res_designation=f"{name}-{chain_id}-{res_id}"
                        if res_designation in catalytic_residues:
                            residues+=[residue]
    
    return residues

def get_catalytic_residues_from_pdb(catalytic_residues,pdb_id,suff=""):
    pdb_file=f"{PDB_FOLDER}/{pdb_id}{suff}.pdb"
    if not os.path.exists(pdb_file):
        return (pdb_id,[])
    structure = parser.get_structure('protein', pdb_file)
    cat_residues={}
    for res_name in catalytic_residues:
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
                            res_designation=f"{name}-{chain_id}-{res_id}"
                            if res_designation in cat_residues[chain_id]:
                                residues+=[(res_designation,residue)]
    
    return (pdb_id,residues)


def get_ligands(pdb_id):
    pdb_file=f"{PDB_FOLDER}/{pdb_id}.pdb"
    structure = parser.get_structure('protein', pdb_file)
    residues={}
    # Loop through all chains in the structure
    for model in structure:
        for chain in model:
            chain_id=chain.get_id()
            # Search for residue by name and position
            for residue in chain:
                if not is_aa(residue):
                    name=residue.get_resname().upper()
                    if name=="HOH":
                        continue
                    res_id=residue.get_id()[1]
                    residues[f"{chain_id}-{name}-{res_id}"]=residue
    
    return residues


def is_close(ligand,residue,th=4.5):
    for atom1 in residue.get_atoms():
        res_coord=atom1.coord
        for atom in ligand.get_atoms():
            dist=np.linalg.norm(atom.coord-res_coord)
            if dist<=th:
                return True
    return False
        

# def extract_residues_by_ids(residue_ids,pdb_id,suff=""):
#     residues=get_residues(pdb_id,suff=suff)
#     return [residues[x] for x in residue_ids]



def check_name(name,res):
    return one_to_three(name)==(res.split("-")[0])

def get_seq(residues_names):
    return "".join([three_to_one(x.split("-")[0]) for x in residues_names])

def get_CA_atom(residue):
    for atom in residue.get_atoms():
        if atom.get_name()=="CA":
            return atom
    return None




def align_residues(pdb_id1,pdb_id2,aln_seq1,aln_seq2):
    assert len(aln_seq1)==len(aln_seq2)
    residues1=get_residues(pdb_id1)
    residues_names1=list(residues1.keys())
    residues2=get_residues(pdb_id2)
    residues_names2=list(residues2.keys())
    mapping={}
    seq1=get_seq(residues_names1)
    seq2=get_seq(residues_names2)
    
    assert seq1 in aln_seq1
    assert seq2 in aln_seq2
    map1=["-"]*len(aln_seq1)
    start1=aln_seq1.index(seq1)
    for i in range(len(seq1)):
        map1[start1+i]=residues_names1[i]
        
    map2=["-"]*len(aln_seq2)
    start2=aln_seq2.index(seq2)
    for i in range(len(seq2)):
        map2[start2+i]=residues_names2[i]
    
    mapping={y:x for x,y in zip(map1,map2) if x!="-" and y!="-"}

    return mapping,residues1,residues2


# Compute RMSD between two sets of residues
def compute_rmsd(residue_set_1, residue_set_2):
    # Extract coordinates of atoms in both residue sets
    coords_1=[]
    for residue in residue_set_1:
        for atom in residue.get_atoms():
            if atom.get_name()=="CA":
                coords_1+=[atom.coord]
                break
    
    
    coords_2=[]
    for residue in residue_set_2:
        for atom in residue.get_atoms():
            if atom.get_name()=="CA":
                coords_2+=[atom.coord]
                break
    
    # Ensure both sets have the same number of atoms
    if len(coords_1) != len(coords_2):
        raise ValueError("Residue sets must have the same number of atoms")

    # Compute RMSD (root mean square deviation)
    coords_1 = np.array(coords_1)
    coords_2 = np.array(coords_2)
    rmsd = np.sqrt(np.mean(np.square(coords_1 - coords_2)))
    
    return rmsd

# # Compute Solvent Accessible Surface Area (SASA) using DSSP
# def compute_sasa(residue_set):
#     structure = parser.get_structure('protein', pdb_file)
#     dssp = DSSP(structure[0], pdb_file)
#     sasa_values = []
    
#     for residue in residue_set:
#         if residue.get_id()[0] == ' ':  # Ensure it's a standard residue (not a heteroatom)
#             res_id = residue.get_id()[1]
#             # Retrieve the DSSP information for the residue
#             if res_id in dssp:
#                 sasa_values.append(dssp[res_id][3])  # DSSP returns a tuple, index 3 is the SASA value
    
#     return sum(sasa_values)


# def compute_sasa_diff(residue_set_1,residue_set_2):
#     sasa_1 = compute_sasa(residue_set_1)
#     sasa_2 = compute_sasa(residue_set_2)

#     print(f'SASA of residue set 1: {sasa_1:.3f} Å²')
#     print(f'SASA of residue set 2: {sasa_2:.3f} Å²')

#     # Compute the difference in SASA
#     return abs(sasa_1 - sasa_2)


def get_uniprot_id(pdb_id):
    try:
        new_pdb_id=pdb_id.lower()
        unps=requests.get(f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{new_pdb_id}").json()[new_pdb_id]['UniProt']
        return list(unps.keys())[0]
    except:
        return "N/A"


import os
import requests

# Define the PDB server URL for ligands


def download_pdb_ligand(ligand_id):
    pdb_url = "https://files.rcsb.org/ligands/download/"
    """
    Download a ligand by its ID from the PDB.
    
    :param ligand_id: Ligand ID to fetch.
    :param save_directory: Directory to save the ligand data file.
    """
    if not os.path.exists(PDB_FOLDER):
        os.makedirs(PDB_FOLDER)
    
    # Construct the URL for downloading the ligand PDB file
    ligand_file_url = f"{pdb_url}{ligand_id}_ideal.sdf"
    
    try:
        # Send GET request to fetch the ligand
        response = requests.get(ligand_file_url)
        response.raise_for_status()
        
        # Save the ligand data to a file
        ligand_file_path = os.path.join(PDB_FOLDER, f"{ligand_id}.sdf")
        with open(ligand_file_path, 'wb') as f:
            f.write(response.content)
        # print(f"Ligand {ligand_id} downloaded successfully.")
    except requests.exceptions.HTTPError as err:
        print(f"Error downloading ligand {ligand_id}: {err}")


import requests

def get_pdb_ids_from_uniprot(uniprot_id):
    try:
        url=f"https://www.ebi.ac.uk/pdbe/graph-api/uniprot/unipdb/{uniprot_id}"

        response = requests.get(url)
        response.raise_for_status()

        result = response.json()
        pdb_ids = [item["name"].lower() for item in result[uniprot_id]["data"]]
        return {uniprot_id:pdb_ids}
    except KeyboardInterrupt:
        import sys
        sys.exit(1)
    except Exception as e:
        # print(e)
        return {uniprot_id:[]}


def get_unp_ids_from_pdb(pdb_id):
    try:
        url=f"https://www.ebi.ac.uk/pdbe/graph-api/mappings/uniprot/{pdb_id}"

        response = requests.get(url)
        response.raise_for_status()

        result = response.json()
        unp_ids = [item for item in result[pdb_id]["UniProt"].keys()]
        return {pdb_id:unp_ids}
    except KeyboardInterrupt:
        import sys
        sys.exit(1)
    except Exception as e:
        # print(e)
        return {pdb_id:[]}


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