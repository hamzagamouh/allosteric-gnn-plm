from extract_ahoj_apo import *

import pandas as pd
from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.Polypeptide import one_to_three,three_to_one,is_aa
import requests
import os

parser=PDBParser()
HOME_FOLDER="/storage/praha1/home/hamzagamouh"
PDB_FOLDER=f"{HOME_FOLDER}/allosteric/pdb_files"
sep="\t"

asd=pd.read_csv(f"{HOME_FOLDER}/allosteric/benchmarks/ASD.csv",sep=sep)

asd=asd[["allosteric_pdb","modulator_alias","modulator_chain","modulator_resi","pdb_uniprot"]].dropna()

with open("ASD_queries.csv","w") as file:
    for i in tqdm(range(asd.shape[0])):
        line=asd.iloc[i,:].to_dict()
        pdb=line["allosteric_pdb"]
        modulators=line["modulator_alias"]
        try:
            float(modulators)
            include=False
        except:
            include=True
        if not include:
            continue
        chains=line["modulator_chain"]
        mod_ids=line["modulator_resi"]
        unp=line["pdb_uniprot"]

        if ";" in chains:
            for chain,modulator,mod_id in zip(chain.split(";"),modulators.split(";"),mod_id.split(";")):
                query=f"{pdb.lower()}{sep}{chain}{sep}{modulator}{sep}{mod_id}{sep}{unp}"
                file.write(f"{query}\n")
        else:
            modulator=modulators
            for chain in chains.split(","):
                chain=chain.strip()
                for mod_id in mod_ids.split("/"):
                    query=f"{pdb.lower()}{sep}{chain}{sep}{modulator}{sep}{mod_id}{sep}{unp}"
                    file.write(f"{query}\n")



# Extract ligands



from Bio.PDB import PDBParser, Select, PDBIO

class LigandSelect(Select):
    def __init__(self, ligand_name, ligand_id):
        self.ligand_name = ligand_name
        self.ligand_id = ligand_id

    def accept_residue(self, residue):
        # Check if residue matches ligand name and ligand id
        resname = residue.get_resname().strip()
        res_id = residue.get_id()[1]  # Residue sequence number
        if resname == self.ligand_name and res_id == self.ligand_id:
            return True
        else:
            return False

# Parameters
pdb_file = "input.pdb"
ligand_name = "LIG"  # Replace with your ligand name
ligand_id = 100      # Replace with your ligand residue number (ID)

# Parse PDB file
parser = PDBParser(QUIET=True)
structure = parser.get_structure("structure", pdb_file)

# Save ligand to a new PDB file
io = PDBIO()
io.set_structure(structure)
io.save("ligand_only.pdb", LigandSelect(ligand_name, ligand_id))

print(f"Ligand {ligand_name} with ID {ligand_id} extracted to ligand_only.pdb")
