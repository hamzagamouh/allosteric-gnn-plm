import numpy as np
from Bio.PDB import PDBParser, PDBIO, Select, is_aa, Polypeptide
from Bio import Align

parser = PDBParser(QUIET=True)
ALIGNER = Align.PairwiseAligner()


class ChainAndLigandSelect(Select):
    def __init__(self, chain_id, lig_name, lig_id):
        self.chain_id = chain_id
        self.lig_name = lig_name
        self.lig_id = lig_id

    def accept_residue(self, residue):
        hetflag, resseq, _ = residue.id
        if hetflag == " " and residue.get_parent().id == self.chain_id:
            return True
        if hetflag.startswith("H_") and residue.resname == self.lig_name and resseq == self.lig_id:
            return True
        return False


def filter_pdb(input_pdb, output_pdb, chain_id, lig_name, lig_id):
    structure = parser.get_structure("structure", input_pdb)
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb, ChainAndLigandSelect(chain_id, lig_name, lig_id))


def get_chain_sequence_from_pdb(filename, pocket):
    pdb_id, chain_id, _, _ = pocket.split("-")
    structure = parser.get_structure(pdb_id, filename)
    chain = structure[0][chain_id]
    residues = [res for res in chain if is_aa(res)]
    try:
        sequence = Polypeptide.Polypeptide(residues).get_sequence()
    except Exception:
        sequence = "".join([Polypeptide.three_to_one(r.get_resname()) for r in residues])
    return residues, str(sequence)


def parse_residue_label(label):
    return label[0], int(label[1:])


def global_alignment(seq1, seq2):
    return ALIGNER.align(seq1, seq2)[0]


def map_residues(alignment, sequence, residues, ref_seq):
    aln_ref_idx, aln_seq_idx = alignment.aligned
    final_residues = [None] * len(ref_seq)
    for (a, b), (c, d) in zip(aln_ref_idx, aln_seq_idx):
        for k_ref, k_seq in zip(range(a, b), range(c, d)):
            final_residues[k_ref] = residues[k_seq]
    return final_residues


def get_ca_atom_coords(residue):
    try:
        return residue["CA"].get_coord() if residue and residue.has_id("CA") else None
    except Exception:
        return None
