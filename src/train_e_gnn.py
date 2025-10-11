import sys
sys.path.append("/storage/brno12-cerit/home/hamzagamouh/.local/lib/python3.8/site-packages")

from e_gnn import GraphRep
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    precision_score, recall_score, matthews_corrcoef
)
import os

import torch 
import torch.nn as nn
import dgl
import pickle
from tqdm import tqdm
import numpy as np

HOME_FOLDER="/storage/praha1/home/hamzagamouh"

MODEL_PATH=f"{HOME_FOLDER}/ahoj-allosteric/src/method_2"

INCLUDE_RES=False

# res_method="plm"
# res_method="fpocket"
res_method="plm"

INCLUDE_GRAPH=True

USE_ATTENTION=True

USE_LIGAND_INFO=False

TRAIN_MODEL=True

print("Including residue features",INCLUDE_RES)
print("Using attention :",USE_ATTENTION)
print("Using ligand info :",USE_LIGAND_INFO)
# Model Definition
class MLP(nn.Module):
    def __init__(self, input_dim, layers_units):
        super(MLP, self).__init__()
        layers = []
        in_dim = input_dim
        if len(layers_units)>0:
            for hidden_dim in layers_units[:-1]:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU())
                # layers.append(nn.Dropout(0.5))
                in_dim = hidden_dim
            hidden_dim=layers_units[-1]
            layers.append(nn.Linear(in_dim, hidden_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv

import dgl
import torch
from dgl.nn.pytorch.conv.egnnconv import EGNNConv

class EGNN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, readout="mean"):
        super(EGNN, self).__init__()
        self.layers = nn.ModuleList()
        self.readout = readout
        self.layers.append(EGNNConv(in_size=in_feats, hidden_size=hidden_feats, out_size=hidden_feats))
        self.layers.append(EGNNConv(in_size=hidden_feats, hidden_size=hidden_feats, out_size=hidden_feats))
        self.layers.append(EGNNConv(in_size=hidden_feats, hidden_size=hidden_feats, out_size=out_feats))

    def forward(self, g):
        h = g.ndata['f']
        x = g.ndata['x']
        for i, layer in enumerate(self.layers):
            h,x = layer(g, h, x)
            if i != len(self.layers) - 1:
                h = F.relu(h)
        
        g.ndata['h'] = h

        hg = dgl.readout_nodes(g, 'h', op='mean')
        return hg

class GCNWithReadout(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, readout="mean"):
        super(GCNWithReadout, self).__init__()
        self.layers = nn.ModuleList()
        self.readout = readout
        self.layers.append(GraphConv(in_feats=in_feats, out_feats=hidden_feats, allow_zero_in_degree=True))
        self.layers.append(GraphConv(in_feats=hidden_feats, out_feats=out_feats, allow_zero_in_degree=True))

    def forward(self, g):
        h = g.ndata['f']
        for i, layer in enumerate(self.layers):
            h= layer(g, h)
            if i != len(self.layers) - 1:
                h = F.relu(h)
        
        g.ndata['h'] = h

        # Graph-level readout
        if self.readout == "sum":
            hg = dgl.readout_nodes(g, 'h', op='sum')
        elif self.readout == "max":
            hg = dgl.readout_nodes(g, 'h', op='max')
        else:  # Default to mean
            hg = dgl.readout_nodes(g, 'h', op='mean')

        return hg


# Model with attention

class GraphListClassifier(nn.Module):
    def __init__(self, in_feats, hidden_dim, rep_dim, res_rep_dim,include_res=INCLUDE_RES,include_graph=INCLUDE_GRAPH):
        super(GraphListClassifier, self).__init__()
        self.include_res=include_res
        self.include_graph=include_graph
        self.in_feats=in_feats
        self.rep_dim=rep_dim
        if self.include_graph:
            # self.graph_rep = GraphRep(in_feats, hidden_dim, rep_dim)
            self.graph_rep = EGNN(in_feats, hidden_dim, rep_dim)
            if USE_ATTENTION:
                # self.graph_rep_keys = EGNN(in_feats, hidden_dim, rep_dim)
                # self.graph_rep_vals = EGNN(in_feats, hidden_dim, rep_dim)
                layers=[2*int(rep_dim)]*3+[rep_dim]
                self.rep_keys = MLP(rep_dim,layers)
                self.rep_vals = MLP(rep_dim,layers)
            # # self.graph_rep_final = GCNWithReadout(rep_dim, hidden_dim, rep_dim)
            self.attn = nn.MultiheadAttention(embed_dim=rep_dim, num_heads=1, batch_first=True)
        if self.include_res:
            res_layers=[512]*2+[128]*2+[64]
            # res_layers=[16]
            self.mlp = MLP(res_rep_dim,res_layers)
        
        if self.include_graph and self.include_res:
            self.classifier = nn.Sequential(
                nn.Linear(rep_dim+res_layers[-1], 128),
                nn.ReLU(),
                # nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.ReLU(),
                # nn.Dropout(0.5),
                nn.Linear(64, 32),
                nn.ReLU(),
                # nn.Dropout(0.5),
                nn.Linear(32, 1)
            )
        elif self.include_graph:
            self.classifier = nn.Sequential(
                nn.Linear(rep_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        elif self.include_res:
            self.classifier = nn.Sequential(
                nn.Linear(res_layers[-1], 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

    def forward_graph(self, batched_graph, use_attention=USE_ATTENTION):
        # Compute graph-level representations
        reps = self.graph_rep(batched_graph).unsqueeze(0)
        if use_attention:
            # Self-attention over graph representations
            keys_reps=self.rep_keys(reps) # Keys 
            vals_reps=self.rep_vals(reps) # Values
            attn_output, vals = self.attn(reps, keys_reps, vals_reps)  # shape: (1, N, rep_dim)
            # Aggregate attended outputs (e.g., mean pooling over all graphs)
            # global_rep = attn_output.max(dim=1).values #+ attn_output.mean(dim=1)  # shape: (1, rep_dim)
            global_rep = attn_output.mean(dim=1) #+ attn_output.mean(dim=1)
        else:
            global_rep=reps.max(dim=1)
        return global_rep



    def forward(self,inputs):
        for k,inp in enumerate(inputs):
            res_emb,graph,ligs=inp
            if self.include_graph:
                graph_rep=self.forward_graph(graph)
                res_rep=graph_rep
            if self.include_res:
                res_emb=self.mlp(res_emb.reshape(1,-1))
                res_rep=res_emb

            if self.include_res and self.include_graph:
                res_rep=torch.cat([res_emb,graph_rep],dim=1)
            # if self.include_res and self.include_graph:
            #     if k==0:
            #         logits=[self.graph_classifier(graph_rep).squeeze(1)+0.1*self.res_classifier(res_emb).squeeze(1)]
            #     else:
            #         logits +=[self.graph_classifier(graph_rep).squeeze(1)+0.1*self.res_classifier(res_emb).squeeze(1)]
            # else:    
            if k==0:
                # Final binary classification
                logits =[self.classifier(res_rep).squeeze(1)]  # shape: (1,)
            else:
                logits +=[self.classifier(res_rep).squeeze(1)]

        logits=torch.cat(logits,dim=0)
        # print(logits)
        # logits=torch.nan_to_num(logits, nan=0.0) 
        # logits=torch.clamp(logits,min=-100,max=100)
        return logits



# class GraphListClassifier(nn.Module):
#     def __init__(self, in_feats, hidden_dim, rep_dim, res_rep_dim,include_res=INCLUDE_RES,include_graph=INCLUDE_GRAPH):
#         super(GraphListClassifier, self).__init__()
#         self.include_res=include_res
#         self.include_graph=include_graph
#         self.in_feats=in_feats
#         self.rep_dim=rep_dim
#         if self.include_graph:
#             # self.graph_rep = GraphRep(in_feats, hidden_dim, rep_dim)
#             self.graph_rep = EGNN(in_feats, hidden_dim, rep_dim)
#             # if USE_ATTENTION:
#             #     self.graph_rep_keys = EGNN(in_feats, hidden_dim, rep_dim)
#             #     self.graph_rep_vals = EGNN(in_feats, hidden_dim, rep_dim)
#             # # self.graph_rep_final = GCNWithReadout(rep_dim, hidden_dim, rep_dim)
#             # self.attn = nn.MultiheadAttention(embed_dim=rep_dim, num_heads=1, batch_first=True)
#         if self.include_res:
#             res_layers=[512]*2+[128]*2+[64]
#             # res_layers=[16]
#             self.mlp = MLP(res_rep_dim,res_layers)
        
#         if self.include_graph and self.include_res:
#             self.classifier = nn.Sequential(
#                 nn.Linear(rep_dim+res_layers[-1], 128),
#                 nn.ReLU(),
#                 nn.Dropout(0.5),
#                 nn.Linear(128, 64),
#                 nn.ReLU(),
#                 nn.Dropout(0.5),
#                 nn.Linear(64, 32),
#                 nn.ReLU(),
#                 nn.Dropout(0.5),
#                 nn.Linear(32, 1)
#             )
#         elif self.include_graph:
#             self.classifier = nn.Sequential(
#                 nn.Linear(rep_dim, 128),
#                 nn.ReLU(),
#                 nn.Linear(128, 64),
#                 nn.ReLU(),
#                 nn.Linear(64, 32),
#                 nn.ReLU(),
#                 nn.Linear(32, 1)
#             )
#         elif self.include_res:
#             self.classifier = nn.Sequential(
#                 nn.Linear(res_layers[-1], 128),
#                 nn.ReLU(),
#                 nn.Linear(128, 64),
#                 nn.ReLU(),
#                 nn.Linear(64, 32),
#                 nn.ReLU(),
#                 nn.Linear(32, 1)
#             )

#     # def forward_graph(self, batched_graph, use_attention=USE_ATTENTION):
#     def forward_graph(self, pocket_graphs):
#         # Compute graph-level representations
#         # apo_graphs = dgl.batch([g for g in dgl.unbatch(batched_graph) if g.ndata['f'][0,-1]==0])
#         # holo_graphs = dgl.batch([g for g in dgl.unbatch(batched_graph) if g.ndata['f'][0,-1]==1])  # each is (1, rep_dim)
#         # reps = torch.cat(reps, dim=0).unsqueeze(0)  # shape: (1, N, rep_dim), batch size = 1
#         max_rep, _ = self.graph_rep(pocket_graphs).unsqueeze(0).max(dim=1) 
#         # mean_rep = self.graph_rep(pocket_graphs).unsqueeze(0).mean(dim=1) 
#         global_rep=max_rep
#         # global_rep=sum_rep
#         return global_rep
    
#     def forward_ligands(self, ligand_feats):
#         # Compute average ligand representations
#         holo_reps = self.mlp(ligand_feats).mean(dim=0).reshape(1,-1)

#         return holo_reps



#     def forward(self,inputs):
#         for k,inp in enumerate(inputs):
#             # res_emb,graph=inp
#             res_emb,pocket_graph,pocket_ligs=inp
#             # print(apo_ligs.shape,holo_ligs.shape)
#             if self.include_graph:
#                 graph_rep=self.forward_graph(pocket_graph)
#                 res_rep=graph_rep
#             if self.include_res:
#                 if not USE_LIGAND_INFO:
#                     res_emb=self.mlp(res_emb.reshape(1,-1))
#                 else:
#                     res_emb=self.forward_ligands(pocket_ligs)
#                 res_rep=res_emb
#             if self.include_res and self.include_graph:
#                 res_rep=torch.cat([res_emb,graph_rep],dim=1)
#             # if self.include_res and self.include_graph:
#             #     if k==0:
#             #         logits=[self.graph_classifier(graph_rep).squeeze(1)+0.1*self.res_classifier(res_emb).squeeze(1)]
#             #     else:
#             #         logits +=[self.graph_classifier(graph_rep).squeeze(1)+0.1*self.res_classifier(res_emb).squeeze(1)]
#             # else:    
#             if k==0:
#                 # Final binary classification
#                 logits =[self.classifier(res_rep).squeeze(1)]  # shape: (1,)
#             else:
#                 logits +=[self.classifier(res_rep).squeeze(1)]

#         logits=torch.cat(logits,dim=0)
#         # print(logits)
#         # logits=torch.nan_to_num(logits, nan=0.0) 
#         # logits=torch.clamp(logits,min=-100,max=100)
#         return logits



import json

import pandas as pd
import pickle
from e_gnn_utils import get_atoms_and_features,build_protein_graph,featurize_sequence


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

import numpy as np
from Bio import PDB
from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB import DSSP
from Bio.PDB.Polypeptide import one_to_three,three_to_one,is_aa
import requests
import os

parser=PDBParser()
PDB_FOLDER=f"{HOME_FOLDER}/allosteric/pdb_files"
SCRATCHDIR=f"{HOME_FOLDER}/allosteric"

def get_catalytic_residues_from_pdb(catalytic_residues,pdb_id,suff=""):
    pdb_file=f"{PDB_FOLDER}/{pdb_id}{suff}.pdb"
    if not os.path.exists(pdb_file):
        print(pdb_file,"don't exist")
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
                            print(res_designation,cat_residues[chain_id])
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
        residues=get_catalytic_residues_from_pdb(new_cat_res,pdb_id)

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


def process_pocket(casbench_pocket):
    graphs=[]
    structures=[]
    for entry in tqdm(get_csa_entries(casbench_pocket)):
        try:
            entry_atoms=get_atoms_from_entry(entry)
        except Exception as e:
            print(e)
            continue
        # print(entry_atoms)
        state=get_state_label(entry["state"])
        for chain_id,residues in entry_atoms.items():
            try:
                atoms, atom_feats, coords=get_atoms_and_features(residues,state)
                graph=build_protein_graph(atom_feats,coords)
                graphs+=[graph]
                structures+=[f"{entry['structure']}-{chain_id}"]
            except Exception as e:
                print(f"Prb with {entry['structure']}-{chain_id}",e)
                pass
    if len(graphs)>0:
        batched_graph=dgl.batch(graphs)
        print(casbench_pocket,structures)
    else:
        batched_graph=[]
    return batched_graph,structures

# for casbench_pocket in tqdm(casbench_pockets):
#     pocket_graph,structures=process_pocket(casbench_pocket)
#     pickle.dump(pocket_graph,open(f"{HOME_FOLDER}/allosteric/method_2/{casbench_pocket}.pkl",'wb'))
#     pickle.dump(structures,open(f"{HOME_FOLDER}/allosteric/method_2/{casbench_pocket}_structures.pkl",'wb'))


df=pd.read_csv(f"{HOME_FOLDER}/allosteric/ahoj_processing/casbench_processed_csa_pockets.csv")

valid_pockets=[x.replace(".pkl","") for x in os.listdir(f"{HOME_FOLDER}/allosteric/method_2/") if ".pkl" in x]

new_df=df[df["NON_CATALYTIC_CSA_POCKET"].isin(valid_pockets)]
allosteric_pockets=df[df["LABEL"]=="Allosteric"]["NON_CATALYTIC_CSA_POCKET"].unique()

all_entries={pocket:entry for pocket,entry in zip(df["NON_CATALYTIC_CSA_POCKET"],df["Entry"])}
dataset=json.load(open(f"{HOME_FOLDER}/allosteric/sequence_dataset.json",'r'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_most_common_count(l):
    count={}
    for x in l:
        if x not in count.keys():
            count[x]=1
        else:
            count[x]+=1

    n_nodes=[y for y in count.keys()]
    freqs=[y for y in count.values()]
    return n_nodes[int(np.argmax(freqs))]


import requests
import requests
from extract_ahoj_apo import *


def extract_ligands(query):
    all_ligs={}
    get_query(query)
    ligands=get_ahoj_ligands(query)
    for info in ligands:
        structure=info["structure"]
        chain_id=info["chains_unp"].upper()
        ligs=["-".join(x.split("_")[1:]) for x in info["pocket_ligs"]]
        all_ligs[f"{structure}-{chain_id}"]=ligs
    delete_query(query)
    return all_ligs

def download_ligand_sdf(ligand_id, filename=None):
    ligand_id = ligand_id.upper()
    url = f"https://files.rcsb.org/ligands/download/{ligand_id}_ideal.sdf"
    if filename is None:
        filename = f"{PDB_FOLDER}/{ligand_id}.sdf"

    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {filename}")
    else:
        print(f"Failed to download {ligand_id}: HTTP {response.status_code}")



from rdkit import Chem

def get_lig_smiles(lig_id):
    filename = f"{PDB_FOLDER}/{lig_id}.sdf"
    # Download if it doesn't exist
    if not os.path.exists(filename):
        print(f"{lig_id} doesn't exist. Downloading...")
        download_ligand_sdf(lig_id)
    suppl = Chem.SDMolSupplier(filename)
    mol = next((m for m in suppl if m is not None), None)  # get first valid molecule
    if mol is None:
        raise ValueError("No valid molecule found in SDF.")
    return Chem.MolToSmiles(mol)  # Canonical SMILES

DPOCKET_FILE=open(f"{HOME_FOLDER}/allosteric/method_2/dpocket_input.txt","w")

def write_dpocket_input(x):
    DPOCKET_FILE.write(f"{x}\n")

def get_allo_ligand(pocket,structure):
    main_query=ahoj_queries[pocket]["query"]
    all_ligs=extract_ligands(main_query)
    if structure in all_ligs.keys():
        ligs=all_ligs[structure]
    else:
        ligs=[]
    if len(ligs)>0:
        try:
            lig_pos=ligs[0]
            lig=lig_pos.split("-")[0]
            write_dpocket_input(f"{structure} \t {lig_pos}")
            lig_smiles=get_lig_smiles(lig)
            lig_feats=featurize_smiles(lig_smiles)
            return lig_feats
        except:
            return [0]*167
    else:
        return [0]*167

# def get_allo_ligand(pocket,structure,pocket_ligand_names):
#     ligs=pocket_ligand_names[structure]
#     if len(ligs)>0:
#         try:
#             lig_pos=ligs[0]
#             lig=lig_pos.split("-")[0]
#             write_dpocket_input(f"{structure} \t {lig_pos}")
#             lig_smiles=get_lig_smiles(lig)
#             lig_feats=featurize_smiles(lig_smiles)
#             return lig_feats
#         except:
#             return [0]*167
#     else:
#         return [0]*167

def get_allo_ligand_name(pocket,structure):
    ligs=[query.split("-")[2] for query in ahoj_queries[pocket] if structure in query]
    if len(ligs)>0:
        return ligs[0]
    else:
        return "apo"

from rdkit.Chem import MACCSkeys

def featurize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    num_atoms = mol.GetNumAtoms()
    # MACCS keys fingerprint returns ExplicitBitVect of length 167 bits
    fp = MACCSkeys.GenMACCSKeys(mol)
    # Convert to numpy array or list of 0/1 integers
    arr = [int(fp.GetBit(i)) for i in range(fp.GetNumBits())]
    return arr
    
def load_data(fold,mode):
    res_feats=0
    inp_feat=0
    dataset_entries=json.load(open(f"{HOME_FOLDER}/allosteric/method_2/casbench_gnn_dataset/{mode}_{fold}.json",'r'))
    data=[]
    # xs=[]
    # ys=[]
    casbench_entries=[]
    for casbench_pocket in tqdm(dataset_entries):
        embs_path=f"{HOME_FOLDER}/allosteric/method_1/non_catalytic_csa"
        fpocket_path=f"{HOME_FOLDER}/allosteric/method_2/fpocket_feats"
        try:
            # if INCLUDE_RES:
            esm_arr=np.load(f"{embs_path}/{casbench_pocket}_esm_embs.npy")
            if esm_arr.shape[0]==0:
                continue
            # fp_arr=np.load(f"{fpocket_path}/{casbench_pocket}_fp_feats.npy")
            # if fp_arr.shape[0]==0:
            #     continue
            if INCLUDE_GRAPH:
                arr=featurize_sequence(dataset[casbench_pocket][0]["sequence"])
                if arr.shape[0]==0 :
                    continue
            labels=np.array([int(x) for x in dataset[casbench_pocket][0]["allosteric_residues"]])
            idx=np.argwhere(labels==1).flatten()
            for i in idx:
                pocket_graph=pickle.load(open(f"{HOME_FOLDER}/allosteric/method_2/{casbench_pocket}.pkl",'rb'))
                pocket_structures=pickle.load(open(f"{HOME_FOLDER}/allosteric/method_2/{casbench_pocket}_structures.pkl",'rb'))
                # pocket_ligands=[get_allo_ligand(casbench_pocket,structure) for structure in pocket_structures]
                # pickle.dump(pocket_ligands,open(f"{HOME_FOLDER}/allosteric/method_2/{casbench_pocket}_ligands.pkl",'wb'))
                pocket_ligands=pickle.load(open(f"{HOME_FOLDER}/allosteric/method_2/{casbench_pocket}_ligands.pkl",'rb'))
                
                orig_graphs=dgl.unbatch(pocket_graph)
                graphs=[]
                ligand_names=[]
                for g,lig_feats in zip(orig_graphs,pocket_ligands):
                    n_nodes=g.ndata['f'].shape[0]
                    lig_feat=torch.tensor([lig_feats]*n_nodes,dtype=torch.float32)
                    g.ndata['x']=torch.cat([g.ndata['f'][:,-4:-1]],dim=1)
                    if USE_LIGAND_INFO:
                        g.ndata['f']=torch.cat([g.ndata['f'][:,:11],lig_feat,g.ndata['f'][:,-1:]],dim=1)
                        # g.ndata['f']=torch.cat([g.ndata['f'][:,:11],g.ndata['f'][:,-1:]],dim=1)
                    else:
                        g.ndata['f']=torch.cat([g.ndata['f'][:,:11],g.ndata['f'][:,-1:]],dim=1)
                    g.ndata['lig']=lig_feat
                    graphs+=[g]
                apo=[g for g in graphs if g.ndata['f'][0,-1]==0]
                holo=[g for g in graphs if g.ndata['f'][0,-1]==1]
                if len(holo)==0:
                    continue
                # n_nodes_apo=[g.ndata['f'].shape[0] for g in apo]
                # n_nodes_holo=[g.ndata['f'].shape[0] for g in holo]
                # best_n_nodes_apo=get_most_common_count(n_nodes_apo)
                # best_n_nodes_holo=get_most_common_count(n_nodes_holo)
                all_n_nodes=[g.ndata['f'].shape[0] for g in graphs]
                best_n_nodes=get_most_common_count(all_n_nodes)
                final_graphs=[g for g in graphs if g.ndata['f'].shape[0]==best_n_nodes]
                final_graphs_idx=[k for k,g in enumerate(graphs) if g.ndata['f'].shape[0]==best_n_nodes]
                # final_graphs=graphs
                apo=[g for g in final_graphs if g.ndata['f'][0,-1]==0]
                holo=[g for g in final_graphs if g.ndata['f'][0,-1]==1]
                # apo=[g for k,g in enumerate(final_graphs) if g.ndata['f'][0,-1]==0 and ligand_names[final_graphs_idx[k]][1]=="apo"]
                # holo=[g for k,g in enumerate(final_graphs) if g.ndata['f'][0,-1]==1 and ligand_names[final_graphs_idx[k]][1]!="apo"]
                # print(casbench_pocket,len(apo),"apo",len(holo),"holo")
                if len(apo)==0:
                    if len(holo)<=1:
                        continue
                else:
                    if len(holo)==0:
                        continue
                pocket_graph=dgl.batch(final_graphs)
                # apo_graphs=dgl.batch(apo)
                # holo_graphs=dgl.batch(holo)
                # apo_graphs=dgl.batch(holo)
                # apo_ligs=torch.cat([g.ndata['lig'][0:1,:] for g in holo],dim=0)
                # holo_ligs=torch.cat([g.ndata['lig'][0:1,:] for g in holo],dim=0)
                ligs=torch.cat([g.ndata['lig'][0:1,:] for g in final_graphs],dim=0)
                n_nodes=pocket_graph.ndata['f'].shape[0]
                if INCLUDE_RES:
                    esm_res_feat=torch.tensor(esm_arr[idx].mean(axis=0),dtype=torch.float32)
                    # fpocket_res_feat=torch.tensor(fp_arr,dtype=torch.float32)
                if INCLUDE_GRAPH:
                    seq_res_feat=torch.tensor([arr[i]]*n_nodes,dtype=torch.float32)
                    
                
                inp_feat=pocket_graph.ndata['f'].shape[-1]
                    
                if INCLUDE_RES:
                    if res_method=="plm":
                        res_feat=esm_res_feat
                    else:
                        res_feat=fpocket_res_feat
                elif INCLUDE_GRAPH:
                    res_feat=seq_res_feat
                # xs=(res_feat.to(device),pocket_graph.to(device))
                xs=(res_feat.to(device),pocket_graph.to(device),ligs.to(device))
                if casbench_pocket in allosteric_pockets:
                    ys=1
                else:
                    ys=0
                if INCLUDE_GRAPH:
                    data+=[(xs,ys)]
                    break
                if INCLUDE_RES:
                    data+=[(xs,ys,casbench_pocket)]
                    break
            casbench_entries+=[all_entries[casbench_pocket]]
        except Exception as e:
            print(e)
            traceback.print_exc()
            continue
    
    print(mode,"-->",len(set(casbench_entries)),"Casbench entries")

    return data,inp_feat


for casbench_pocket in tqdm(valid_pockets):
    pocket_graph=pickle.load(open(f"{HOME_FOLDER}/allosteric/method_2/{casbench_pocket}.pkl",'rb'))
    break

def print_stats(data):
    y=np.array([item[1] for item in data])
    print("Positives",(y==1).sum())
    print("Negatives",(y==0).sum())



# Download all ligs

# all_ligs=[]
# for file in os.listdir(f"{HOME_FOLDER}/allosteric/method_2/"):
#     if "_structure" in file: 
#         pocket_structures=pickle.load(open(f"{HOME_FOLDER}/allosteric/method_2/{file}",'rb'))
#         casbench_pocket=file.replace("_structures.pkl","")
        # all_ligs+=[get_allo_ligand(casbench_pocket,structure) for structure in pocket_structures]

# all_ligs=[x for x in set(all_ligs) if x!="apo"]

# for lig in all_ligs:
#     download_ligand_sdf(lig)
# import sys
# sys.exit(0)

for fold in range(5):
    print("FOLD",fold)
    # if fold==0:
    #     TRAIN_MODEL=False
    # else:
    #     TRAIN_MODEL=True

    train_data,inp_feat=load_data(fold,mode="train")
    print("Training",len(train_data))
    print_stats(train_data)
    val_data,inp_feat=load_data(fold,mode="val")
    print("Val",len(val_data))
    print_stats(val_data)

    # train_data=train_data+val_data
    # print("Training",len(train_data))
    # print_stats(train_data)


    test_data,inp_feat=load_data(fold,mode="test")
    print("Test",len(test_data))
    print_stats(test_data)

    # x_train,y_train,res_feats=load_data(fold,mode="train")
    # print("Training",len(x_train))
    # x_val,y_val,res_feats=load_data(fold,mode="val")
    # print("Val",len(x_val))
    # x_test,y_test,res_feats=load_data(fold,mode="test")
    # print("Test",len(x_test))

    # print("Residue features",inp_feat)

    if INCLUDE_RES:
        if res_method=="plm":
            if USE_LIGAND_INFO:
                res_feats=167 # Ligands
            else:
                res_feats=2560 # ESM embeddings
        else:
            res_feats=19 # Fpocket
    else:
        res_feats=21  # Onehot

    # hidden=256
    # output=1024
    hidden=128
    output=128
    print("input feats",inp_feat)
    model=GraphListClassifier(inp_feat,hidden,output,res_feats).to(device)

    # model.load_state_dict(torch.load("GNN_model.pth"))


    epochs = 1000
    lr = 1e-4
    y_train=np.array([item[1] for item in train_data])
    # pos_weight = torch.tensor([((y_train == 0).sum() / (y_train == 1).sum())], dtype=torch.float32).to(device)
    # # pos_weight = torch.tensor([1], dtype=torch.float32).to(device)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    batch_size=16 #128
    # # Convert to PyTorc
    import random


    def shuffle_data(data):
        idx=list(range(len(data)))
        random.shuffle(idx)
        return [data[i] for i in idx]

    # Training loop
    def evaluate(data):
        model.eval()
        preds = []
        targets = []
        with torch.no_grad():
            for a in tqdm(range(0,len(data),batch_size),desc="Evaluating"):
                b=a+batch_size
                # xb=[(item[0][0].to(device),item[0][1].to(device)) for item in data[a:b]]
                xb=[(item[0][0],item[0][1],item[0][2]) for item in data[a:b]]
                yb=torch.tensor([item[1] for item in data[a:b]],dtype=torch.float32).to(device)
                logits = model(xb)
                probs = torch.sigmoid(logits)
                preds.append(probs.cpu().numpy())
                targets.append(yb.cpu().numpy())
                
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        # print("preds",preds)
        # print("targets",targets)
        pred_labels = (preds > 0.5).astype(int)
        return {

            "accuracy": accuracy_score(targets, pred_labels),
            "auc": roc_auc_score(targets, preds),
            "aupr": average_precision_score(targets, preds),
            "precision": precision_score(targets, pred_labels),
            "recall": recall_score(targets, pred_labels),
            "mcc": matthews_corrcoef(targets, pred_labels)
        }




    model_name=""
    if INCLUDE_RES:
        model_name+=res_method
    if INCLUDE_GRAPH:
        model_name+="GNN"

    metric="auc"
    
    print("Model",model_name)
    if TRAIN_MODEL:
        max_val=0
        max_train=0
        for epoch in tqdm(range(1, epochs + 1),desc="Training"):
            model.train()
            running_loss = 0.0
            total_loss=torch.tensor(0,dtype=torch.float32).to(device)
            for a in (range(0,len(train_data),batch_size)):
                # for name, param in model.named_parameters():
                #     if param.requires_grad:
                #         print(f"Weights : {name} mean: {param.data.mean():.4f}, std: {param.data.std():.4f}")
                #         if param.grad is not None:
                #             print(f"{name} grad mean: {param.grad.mean():.6f}, std: {param.grad.std():.6f}")
                #         else:
                #             print(f"{name} has no grad")
                b=a+batch_size
                xb=[(item[0][0],item[0][1],item[0][2]) for item in train_data[a:b]]
                yb=torch.tensor([item[1] for item in train_data[a:b]],dtype=torch.float32).to(device)
                pos_weight = torch.tensor([((yb == 0).sum() / (yb == 1).sum())], dtype=torch.float32).to(device)
                if pos_weight.item()>1e5:
                    pos_weight=torch.tensor([1], dtype=torch.float32).to(device)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            train_loss = running_loss / len(train_data)
            train_data=shuffle_data(train_data)

            if epoch%10==0:
                print("Epoch ",epoch)
                train_metrics = evaluate(train_data)
                val_metrics = evaluate(val_data)

                if val_metrics[metric]>max_val and train_metrics[metric]>val_metrics[metric]:
                    print('Saving model...')
                    # torch.save(model.state_dict(), f"{metric}_models/{model_name}_model_fold_{fold}.pth")
                    if INCLUDE_RES:
                        torch.save(model.state_dict(),f"{MODEL_PATH}/{metric}_models/{res_method}/{model_name}_model_fold_{fold}.pth")
                    elif INCLUDE_GRAPH:
                        if USE_ATTENTION:
                            if USE_LIGAND_INFO:
                                torch.save(model.state_dict(),f"{MODEL_PATH}/{metric}_models/attention_ligand_info/{model_name}_model_fold_{fold}.pth")
                            else:
                                torch.save(model.state_dict(),f"{MODEL_PATH}/{metric}_models/attention/{model_name}_model_fold_{fold}.pth")
                        else:
                            torch.save(model.state_dict(),f"{MODEL_PATH}/{metric}_models/no_attention/{model_name}_model_fold_{fold}.pth")
                    max_val=val_metrics[metric]

                print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Train {metric}: {train_metrics[metric]:.4f} | Val {metric}: {val_metrics[metric]:.4f}")

            
        


    # Final Evaluation on Test Set
    try:
        if INCLUDE_RES:
            if res_method=="plm":
                # res_feats=2560
                res_feats=167
            else:
                res_feats=19
        else:
            res_feats=21  # 2560 or 21
        model=GraphListClassifier(inp_feat,hidden,output,res_feats).to(device)
        print("Loading model...")
        if INCLUDE_RES:
            model.load_state_dict(torch.load(f"{MODEL_PATH}/{metric}_models/{res_method}/{model_name}_model_fold_{fold}.pth"))
        elif INCLUDE_GRAPH:
            if USE_ATTENTION:
                if USE_LIGAND_INFO:
                    model.load_state_dict(torch.load(f"{MODEL_PATH}/{metric}_models/attention_ligand_info/{model_name}_model_fold_{fold}.pth"))
                else:
                    model.load_state_dict(torch.load(f"{MODEL_PATH}/{metric}_models/attention/{model_name}_model_fold_{fold}.pth"))
            else:
                model.load_state_dict(torch.load(f"{MODEL_PATH}/{metric}_models/no_attention/{model_name}_model_fold_{fold}.pth"))
    except Exception as e:
        print("Error",e)



    train_metrics = evaluate(train_data)
    val_metrics = evaluate(val_data)
    test_metrics = evaluate(test_data)
    print("\n--- Final Evaluation ---")
    for mode,metric in zip(["train","val","test"],[train_metrics,val_metrics,test_metrics]):
        print(mode,"metrics")
        for k, v in metric.items():
            print(f"{k.capitalize():10s}: {v:.4f}")
