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
import random

import torch 
import torch.nn as nn
import dgl
import pickle
from tqdm import tqdm
import numpy as np

HOME_FOLDER="/storage/praha1/home/hamzagamouh"

MODEL_PATH=f"{HOME_FOLDER}/ahoj-allosteric/src/method_2"

INCLUDE_RES=False

res_method="plm"

INCLUDE_GRAPH=True

USE_ATTENTION=True

USE_LIGAND_INFO=False

TRAIN_MODEL=True

print("Using attention :",USE_ATTENTION)
print("Using ligand info :",USE_LIGAND_INFO)
print("dpocket feats")
# Model Definition

class MLP(nn.Module):
    def __init__(self, input_dim, layers_units,dropout=False):
        super(MLP, self).__init__()
        layers = []
        in_dim = input_dim
        for hidden_dim in layers_units[:-1]:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if dropout:
                layers.append(nn.Dropout(0.5))
            in_dim=hidden_dim
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
from dgl.nn.pytorch.conv import AtomicConv 
from dgl.nn.pytorch.conv.egatconv import EGATConv


# Model with attention

class GraphListClassifier(nn.Module):
    def __init__(self, in_feats, hidden_dim, rep_dim,include_res=INCLUDE_RES,include_graph=INCLUDE_GRAPH):
        super(GraphListClassifier, self).__init__()
        self.include_res=include_res
        self.include_graph=include_graph
        self.in_feats=in_feats
        self.rep_dim=rep_dim
        if self.include_graph:
            # layers=[2*int(rep_dim)]*3+[rep_dim]
            layers=[2*int(rep_dim)]*3+[rep_dim]
            self.graph_rep = MLP(in_feats, layers,dropout=False)
            if USE_ATTENTION:
                self.attn = nn.MultiheadAttention(embed_dim=rep_dim, num_heads=1, batch_first=True)
        
        
        if self.include_graph:
            self.classifier = nn.Sequential(
                nn.Linear(int(rep_dim), 1),
                # nn.Linear(int(rep_dim), 128),
                # nn.ReLU(),
                # nn.Linear(128, 64),
                # nn.ReLU(),
                # nn.Linear(64, 32),
                # nn.ReLU(),
                # nn.Linear(32, 1),
                # nn.Softmax()
            )
    

    def forward_graph(self, graph_reps, use_attention=USE_ATTENTION):
        # Compute graph-level representations
        reps=self.graph_rep(graph_reps).unsqueeze(0)
        if use_attention:
            attn_output, vals = self.attn(reps, reps, reps)  # shape: (1, N, rep_dim)
            # Aggregate attended outputs (e.g., mean pooling over all graphs)
            # print(attn_output.shape,vals.shape,torch.argmax(vals,dim=1))
            global_rep = attn_output.max(dim=1).values #+ attn_output.mean(dim=1)
        else:
            # Geometric features 
            global_rep=reps.max(dim=1).values
        return global_rep



    def forward(self,inputs):
        for k,inp in enumerate(inputs):
            reps=inp
            graph_rep=self.forward_graph(reps)
            res_rep=graph_rep
            if k==0:
                # Final binary classification
                logits =[self.classifier(res_rep)]  # shape: (1,)
            else:
                logits +=[self.classifier(res_rep)]

        logits=torch.cat(logits,dim=0).reshape(-1)#.softmax(dim=1)
        return logits


import json

import pandas as pd
import pickle
from e_gnn_utils import get_atoms_and_features,build_protein_graph,featurize_sequence


pocket_mapping=json.load(open(f"{HOME_FOLDER}/allosteric/ahoj_processing/pocket_mapping.json",'r'))



dpocket_df=pd.read_csv(f"{HOME_FOLDER}/allosteric/method_2/dpocket_feats/all_outputs.txt")
cols=[x for x in dpocket_df.columns if x not in ["pdb","lig"]]

def normalize_entry(x):
    return "-".join(x.split("_"))

all_dpocket_feats={normalize_entry(x):y for x,y in zip(dpocket_df.loc[:,"pdb"],dpocket_df.loc[:,cols].values)}


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


def process_catalytic_pocket(casbench_pocket):
    graphs=[]
    structures=[]
    k=0
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
                if k==0:
                    ref_residues=residues
                atoms, atom_feats, coords=get_atoms_and_features(residues,ref_residues,state)
                graph=build_protein_graph(atom_feats,coords)
                graphs+=[graph]
                structures+=[f"{entry['structure']}-{chain_id}"]
                k+=1
            except Exception as e:
                print(f"Prb with {entry['structure']}-{chain_id}",e)
                pass
    if len(graphs)>0:
        batched_graph=dgl.batch(graphs)
        print(casbench_pocket,structures)
    else:
        batched_graph=[]
    return batched_graph,structures


def process_allosteric_pocket(allo_res):
    graphs=[]
    k=0
    for structure,residue_names in allo_res.items():
        pdb_id,chain_id=structure.split("-")
        residues=[x[1] for x in get_allo_res(pdb_id,residue_names)]
        state=0
        try:
            if k==0:
                ref_residues=residues
            atoms, atom_feats, coords=get_atoms_and_features(residues,ref_residues,state)
            graph=build_protein_graph(atom_feats,coords)
            graphs+=[graph]
            k+=1
        except Exception as e:
            print(f"Prb with {structure}-{chain_id}",e)
            pass
    if len(graphs)>0:
        print(f"Found {len(graphs)} graphs")
        batched_graph=dgl.batch(graphs)
    else:
        batched_graph=[]
    return batched_graph

dataset=json.load(open(f"{HOME_FOLDER}/allosteric/sequence_dataset.json",'r'))

def unify_res_names(x):
    if "_" in x:
        chain_id,pos=x.split("_")
        return f"XXX-{chain_id}-{pos}"
    return x

# for casbench_pocket in tqdm(casbench_pockets):
#     pocket_graph,structures=process_catalytic_pocket(casbench_pocket)
# #     allo_res={}
# #     for d in dataset[casbench_pocket]:
# #         allo_res[f"{d['pdb_id']}-{d['chain_id']}"]=[unify_res_names(x) for x in d['allosteric_names']]
# #     allo_pocket_graph=process_allosteric_pocket(allo_res)
#     pickle.dump(pocket_graph,open(f"{HOME_FOLDER}/allosteric/method_2/{casbench_pocket}.pkl",'wb'))
#     pickle.dump(structures,open(f"{HOME_FOLDER}/allosteric/method_2/{casbench_pocket}_structures.pkl",'wb'))

df=pd.read_csv(f"{HOME_FOLDER}/allosteric/ahoj_processing/casbench_processed_csa_pockets.csv")

valid_pockets=[x.replace(".pkl","") for x in os.listdir(f"{HOME_FOLDER}/allosteric/method_2/") if ".pkl" in x]

new_df=df[df["NON_CATALYTIC_CSA_POCKET"].isin(valid_pockets)]
allosteric_pockets=df[df["LABEL"]=="Allosteric"]["NON_CATALYTIC_CSA_POCKET"].unique()

all_entries={pocket:entry for pocket,entry in zip(df["NON_CATALYTIC_CSA_POCKET"],df["Entry"])}


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

import dgl
import torch

def set_adjacency_matrix(g, adj,edge_dists):
    """
    Replace the graph's structure with a new adjacency matrix.
    adj: a (N, N) binary tensor (0 or 1), where adj[i, j] = 1 means edge i→j
    """
    assert adj.dim() == 2 and adj.size(0) == adj.size(1), "adj must be a square matrix"
    num_nodes = adj.size(0)

    # Get edge indices where adj[i, j] == 1
    src, dst = torch.nonzero(adj, as_tuple=True)

    # Create a new graph with the same number of nodes but new edges
    new_g = dgl.graph((src, dst), num_nodes=num_nodes)

    for key, feat in g.ndata.items():
        new_g.ndata[key] = feat.clone()
    
    # new_g.edata["dist"] = edge_dists

    return new_g


import torch

def superimpose_atoms_torch(A, B):
    """
    Superimpose set B onto set A using the Kabsch algorithm in PyTorch.
    A and B are (N, 3) tensors of corresponding atom coordinates.
    
    Returns:
        B_aligned: (N, 3) aligned version of B
        R: (3, 3) rotation matrix
        t: (3,) translation vector
    """
    assert A.shape == B.shape
    # A = A.double()
    # B = B.double()
    
    # 1. Compute centroids
    centroid_A = A.mean(dim=0).reshape(1,-1)
    centroid_B = B.mean(dim=0).reshape(1,-1)
    
    # 2. Center the coordinates
    A_centered = A - centroid_A
    B_centered = B - centroid_B

    # 3. Compute covariance matrix
    H = B_centered.T @ A_centered  # (3, N) @ (N, 3) -> (3, 3)

    # 4. SVD
    U, S, Vt = torch.linalg.svd(H)
    R = (Vt.T @ U.T).T
    # 5. Correct for reflection
    if torch.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = (Vt.T @ U.T).T

    # 6. Compute translation
    t = centroid_A - centroid_B @ R
    # 7. Apply rotation and translation to B
    # B_aligned = (R @ B.T).T + t
    B_aligned = B @ R + t

    return B_aligned, R, t



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
    suppl = Chem.SDMolSuppplier(filename)
    mol = next((m for m in suppl if m is not None), None)  # get first valid molecule
    if mol is None:
        raise ValueError("No valid molecule found in SDF.")
    return Chem.MolToSmiles(mol)  # Canonical SMILES

# DPOCKET_FILE=open(f"{HOME_FOLDER}/allosteric/method_2/dpocket_input.txt","w")

def write_dpocket_input(x):
    with open(f"{PDB_FOLDER}/dpocket_input.txt","a") as f:
        f.write(f"{x}\n")

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
            write_dpocket_input(f"{structure.split('-')[0]}.pdb\t{lig}")
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
    

def graphs_have_same_connectivity(g1, g2):
    # Get edge lists as sets of tuples
    edges1 = set(zip(g1.edges()[0].tolist(), g1.edges()[1].tolist()))
    edges2 = set(zip(g2.edges()[0].tolist(), g2.edges()[1].tolist()))

    return edges1 == edges2



def pairwise_euclidean_distances(x):
    """
    Compute Euclidean distance matrix for x (N x D).
    Returns:
        dist_matrix (N x N): pairwise distances
    """
    x_norm = (x ** 2).sum(dim=1).unsqueeze(1)  # (N, 1)
    dist_squared = x_norm + x_norm.T - 2.0 * torch.matmul(x, x.T)
    dist_squared = torch.clamp(dist_squared, min=0.0)  # Avoid negative values due to numerical error
    distances = torch.sqrt(dist_squared)
    return distances

def get_state(g):
    return g.ndata['f'][0:1,-1].numpy().reshape(1,1)

def load_data(fold,mode):
    # dataset_entries=json.load(open(f"{HOME_FOLDER}/allosteric/method_2/casbench_gnn_dataset/{mode}_{fold}.json",'r'))
    dataset_entries=json.load(open(f"{HOME_FOLDER}/allosteric/method_1/casbench_enriched_dataset/{mode}_{fold}.json",'r'))
    dataset_entries=[x for x in dataset_entries if x in casbench_pockets]
    data=[]
    # xs=[]
    # ys=[]
    casbench_entries=[]
    for casbench_pocket in tqdm(dataset_entries):
        embs_path=f"{HOME_FOLDER}/allosteric/method_1/non_catalytic_csa"
        fpocket_path=f"{HOME_FOLDER}/allosteric/method_2/fpocket_feats"
        try:
            # if INCLUDE_RES:
            # print("Loading PLM embs")
            esm_arr=np.load(f"{embs_path}/{casbench_pocket}_esm_embs.npy")
            if esm_arr.shape[0]==0:
                continue
            seq_arr=featurize_sequence(dataset[casbench_pocket][0]["sequence"])
            if seq_arr.shape[0]==0 :
                continue
            # print("Getting labels")
            labels=np.array([int(x) for x in dataset[casbench_pocket][0]["allosteric_residues"]])
            idx=np.argwhere(labels==1).flatten()
            for i in idx:
                # print("Loading catalytic graphs")
                one_hot_feats=np.max(seq_arr[idx],axis=0).reshape(1,-1)
                esm_feats=esm_arr[idx].mean(axis=0).reshape(1,-1)
                pocket_graph=pickle.load(open(f"{HOME_FOLDER}/allosteric/method_2/{casbench_pocket}.pkl",'rb'))
                pocket_structures=pickle.load(open(f"{HOME_FOLDER}/allosteric/method_2/{casbench_pocket}_structures.pkl",'rb'))
                pocket_ligands=pickle.load(open(f"{HOME_FOLDER}/allosteric/method_2/{casbench_pocket}_ligands.pkl",'rb'))
                # lig_feats=np.array(pocket_ligands).max(axis=0).reshape(1,-1)
                lig_feats=np.array(pocket_ligands)
                # dpocket_feats=[np.concatenate([arr],axis=1) for x in pocket_structures if x in all_dpocket_feats.keys()]
                # Apo/Holo states
                # graphs=dgl.unbatch(pocket_graph)
                # dpocket_feats=[np.concatenate([all_dpocket_feats[x].reshape(1,-1),esm_feats,get_state(g)],axis=1) for x,g in zip(pocket_structures,graphs) if x in all_dpocket_feats.keys()]
                # if len(dpocket_feats)==0:
                #     continue
                # all_feats=np.concatenate(dpocket_feats,axis=0)
                n_structures=lig_feats.shape[0]
                # seq_feats=np.concatenate([seq_arr[idx].max(axis=0).reshape(1,-1)]*n_structures,axis=0)
                seq_feats=np.concatenate([esm_arr[idx].max(axis=0).reshape(1,-1)]*n_structures,axis=0)
                all_feats=np.concatenate([lig_feats,seq_feats],axis=1)
                reps=torch.tensor(all_feats,dtype=torch.float32)
                inp_feat=reps.shape[-1]
                xs=(reps.to(device))
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

for fold in range(10):
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

    # x_train,y_train,res_feats=load_data(fold,mode="train")
    # print("Training",len(x_train))
    # x_val,y_val,res_feats=load_data(fold,mode="val")
    # print("Val",len(x_val))
    # x_test,y_test,res_feats=load_data(fold,mode="test")
    # print("Test",len(x_test))

    # print("Residue features",inp_feat)

    # hidden=256
    hidden=1024
    output=1024
    # hidden=128
    # output=128
    print("input feats",inp_feat)
    model=GraphListClassifier(inp_feat,hidden,output).to(device)

    # model.load_state_dict(torch.load("GNN_model.pth"))


    epochs = 100
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
                # xb=[(item[0][0],item[0][1],item[0][2]) for item in data[a:b]]
                xb=[(item[0]) for item in data[a:b]]
                yb=torch.tensor([item[1] for item in data[a:b]],dtype=torch.float32).to(device)
                logits = model(xb)
                probs = torch.sigmoid(logits)
                # probs  = model(xb)[:,1]
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
                b=a+batch_size
                # xb=[(item[0][0],item[0][1],item[0][2]) for item in train_data[a:b]]
                xb=[(item[0]) for item in train_data[a:b]]
                yb=torch.tensor([item[1] for item in train_data[a:b]],dtype=torch.float32).to(device)
                n0=(yb == 0).sum()
                n1=(yb == 1).sum()
                if n1>0 and n0>0:
                    w1=n0/n1
                else:
                    w1=torch.tensor([1],dtype=torch.float32).to(device)
                criterion = nn.BCEWithLogitsLoss(pos_weight=w1)
                # w1 = n0/(n0+n1)
                # w0 = n1/(n0+n1)
                # weight=(torch.Tensor([w0,w1])).to(device)
                # criterion=nn.CrossEntropyLoss(weight)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb.reshape(-1,).float())
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            train_loss = running_loss / len(train_data)
            train_data=shuffle_data(train_data)

            if epoch%10==0:
                print("Epoch ",epoch)
                train_metrics = evaluate(train_data)
                val_metrics = evaluate(val_data)

                # if val_metrics[metric]>=max_val and train_metrics[metric]>=max_train and train_metrics[metric]>=val_metrics[metric]:
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
                max_train=train_metrics[metric]

                print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Train {metric}: {train_metrics[metric]:.4f} | Val {metric}: {val_metrics[metric]:.4f}")

            
        

test_data,inp_feat=load_data("final",mode="test")
print("Test",len(test_data))
print_stats(test_data)

# Final Evaluation on Test Set
try:
    if INCLUDE_RES:
        if res_method=="plm":
            res_feats=2560
            # res_feats=167
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


test_metrics = evaluate(test_data)
print("\n--- Final Evaluation ---")
for mode,metric in zip(["test"],[test_metrics]):
    print(mode,"metrics")
    for k, v in metric.items():
        print(f"{k.capitalize():10s}: {v:.4f}")


