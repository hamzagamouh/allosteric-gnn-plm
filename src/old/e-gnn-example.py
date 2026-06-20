import sys
sys.path.append("/storage/brno12-cerit/home/hamzagamouh/.local/lib/python3.8/site-packages")

from e_gnn import GraphRep

# model=GraphRep(num_input_feats=3, num_hidden_feats=256,
#                      num_output_feats=128, num_edge_input_feats=0)

import torch 
import torch.nn as nn
import dgl
import pickle
from tqdm import tqdm

HOME_FOLDER="/storage/praha1/home/hamzagamouh"



class GraphListClassifier(nn.Module):
    def __init__(self, in_feats, hidden_dim, rep_dim):
        super(GraphListClassifier, self).__init__()
        self.graph_rep = GraphRep(in_feats, hidden_dim, rep_dim)
        self.attn = nn.MultiheadAttention(embed_dim=rep_dim, num_heads=1, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(rep_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, batched_graph):
        # Compute graph-level representations
        # reps = [self.graph_rep(g) for g in graph_list]  # each is (1, rep_dim)
        # reps = torch.cat(reps, dim=0).unsqueeze(0)  # shape: (1, N, rep_dim), batch size = 1
        reps = self.graph_rep(batched_graph).unsqueeze(0)
        # Self-attention over graph representations
        attn_output, vals = self.attn(reps, reps, reps)  # shape: (1, N, rep_dim)

        # print(vals.shape,attn_output.shape)
        # Aggregate attended outputs (e.g., mean pooling over all graphs)
        global_rep = attn_output.mean(dim=1)  # shape: (1, rep_dim)

        # Final binary classification
        logits = self.classifier(global_rep)  # shape: (1, 1)
        return torch.sigmoid(logits).squeeze(1)  # shape: (1,)

import json

import pandas as pd
import pickle
from e_gnn_utils import get_atoms_and_features,build_protein_graph


pocket_mapping=json.load(open(f"{HOME_FOLDER}/allosteric/ahoj_processing/pocket_mapping.json",'r'))



new_df=pd.read_csv(f"{HOME_FOLDER}/allosteric/ahoj_processing/casbench_processed_csa_pockets.csv")

print(len(new_df[new_df["NON_CATALYTIC_CSA_POCKET"]!="N/A"]["Entry"].unique()),"casbench entries")


non_allo_casbench_pockets=new_df[new_df["LABEL"]=="Non_Allosteric"]["NON_CATALYTIC_CSA_POCKET"].unique()
allo_casbench_pockets=new_df[new_df["LABEL"]=="Allosteric"]["NON_CATALYTIC_CSA_POCKET"].unique()

def get_csa_entries(casbench_pocket):
    entries=[]
    for entry in pocket_mapping:
        if entry["pocket"]==casbench_pocket:
            entries+=[entry]

    return entries




SCRATCHDIR=f"{HOME_FOLDER}/allosteric"

def get_atoms_from_entry(entry):
    pdb_id=entry["structure"]

    all_residues={pdb_id:pickle.load(open(f"{SCRATCHDIR}/rmsds/all_residues_{pdb_id}.res","rb"))}
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


def process_pocket(casbench_pocket):
    graphs=[]
    for entry in tqdm(get_csa_entries(casbench_pocket)):
        entry_atoms=get_atoms_from_entry(entry)
        for chain_id,residues in entry_atoms.items():
            atoms, atom_feats, coords=get_atoms_and_features(residues)
            graph=build_protein_graph(coords,coords)
            graph.ndata["state"]=torch.tensor([get_state_label(entry["state"]) for _ in range(len(graph.ndata["x"]))])
            graphs+=[graph]
    batched_graph=dgl.batch(graphs)
    return batched_graph

for casbench_pocket in allo_casbench_pockets:
    pocket_graph=(casbench_pocket)
    break



model=GraphListClassifier(3,256,128)


