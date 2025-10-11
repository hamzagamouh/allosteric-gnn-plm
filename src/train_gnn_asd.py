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
    precision_score, recall_score, matthews_corrcoef, f1_score
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

TRAIN_MODEL=True

# Model architecture
import argparse

# Initialize the parser
parser = argparse.ArgumentParser()

# Add a string argument called 'archi'
parser.add_argument('--archi', type=str, required=True, help='Architecture name',default="seq+geom+lig")

# Parse arguments
args = parser.parse_args()

ARCHI=args.archi



class MLP(nn.Module):
    def __init__(self, input_dim, layers_units,dropout=False):
        super(MLP, self).__init__()
        layers = []
        in_dim = input_dim
        for hidden_dim in layers_units[:-1]:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(0.9))
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


class GCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, readout="mean"):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.readout = readout

        # Graph Conv
        self.layers.append(GraphConv(in_feats=in_feats, out_feats=hidden_feats, allow_zero_in_degree=True))
        self.layers.append(GraphConv(in_feats=hidden_feats, out_feats=hidden_feats, allow_zero_in_degree=True))
        self.layers.append(GraphConv(in_feats=hidden_feats, out_feats=out_feats, allow_zero_in_degree=True))
  
    def forward(self, g):
        h = g.ndata['feat']
        for i, layer in enumerate(self.layers):
            # GraphConv
            h= layer(g, h)#, edge_weight=g.edata['dist'])

            h=h.reshape(h.shape[0],-1)
            if i != len(self.layers) - 1:
                h = F.relu(h)
            
        
        g.ndata['h'] = h

        return g


# Model with attention
ESM_DIM=2560

class GraphListClassifier(nn.Module):
    def __init__(self, in_feats, hidden_dim, rep_dim):
        super(GraphListClassifier, self).__init__()
        self.in_feats=in_feats
        self.rep_dim=rep_dim
        self.graph_rep = GCN(in_feats, hidden_dim, rep_dim)
        self.query=MLP(rep_dim,[64]*3+[rep_dim])
        self.key=MLP(rep_dim,[64]*3+[rep_dim])
        self.value=MLP(rep_dim,[64]*3+[rep_dim])
        self.attn = nn.MultiheadAttention(embed_dim=rep_dim, num_heads=1, batch_first=True)
        dim=int(rep_dim)
        self.classifier = nn.Sequential(
            # nn.Linear(dim+ESM_DIM, dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

    def forward_graph(self, batched_graph,use_attention=True):
        # Compute graph-level representations
        gs=dgl.unbatch(self.graph_rep(batched_graph))
        hs=[g.ndata["h"] for g in gs]
        # seq_feats=gs[0].ndata["seq_feat"]
        hs=torch.cat([h.reshape(h.shape[0],1,h.shape[1]) for h in hs],dim=1)
        reps=hs
        if use_attention:
            queries=self.query(reps)
            keys=self.key(reps)
            values=self.value(reps)
            attn_output, vals = self.attn(queries, keys, values)  # shape: (1, N, rep_dim)
            # Aggregate attended outputs (e.g., mean/max pooling over all graphs)
            global_rep = attn_output.mean(dim=1)
        else:
            # Geometric features 
            global_rep=reps.max(dim=1)[0]
        # return torch.cat([global_rep,seq_feats],dim=1)
        return global_rep

    def forward(self,inputs):
        logits=[]
        for gs in inputs:
            logit=self.classifier(self.forward_graph(gs))
            logits+=[logit]

        return torch.cat(logits,dim=0).reshape(-1)


import json

import pandas as pd
import pickle
from e_gnn_utils import get_atoms_and_features,build_protein_graph,featurize_sequence
import dgl



asd_dataset=json.load(open(f"{HOME_FOLDER}/allosteric/method_2/ASD_dataset.json","r"))
asd_entries=json.load(open(f"{HOME_FOLDER}/ahoj-allosteric/src/method_2/ASD_entries.json","r"))

maccs_fps=pickle.load(open(f"{HOME_FOLDER}/allosteric/asd_processing/ligands_maccs.pkl",'rb'))

def load_esm_embs(unp):
    path=f"{HOME_FOLDER}/allosteric/method_2/ASD_{unp}_esm_embs.npy"
    if os.path.exists(path):
        return np.load(path)
    else:
        return None

def load_ligand_info(unp):
    ligand_info=json.load(open(f"{HOME_FOLDER}/allosteric/asd_processing/ligand_info/{unp}_ligands.json",'r'))
    
    for entry,info in ligand_info.items():
        for lig,labels in info.items():
            n_res=len(labels)
        #     break
        # break

    lig_feats=np.zeros((n_res,167))
    
    for entry,info in ligand_info.items():
        for lig,labels in info.items():
            n_res=len(labels)
            lig_name=lig.split("-")[0]
            lig_fps=maccs_fps.get(lig_name,None)
            for k in range(len(labels)):
                if labels[k]=="1":
                    lig_feats[k]+=np.array(lig_fps)
                    
    lig_feats=(lig_feats>0).astype(np.float32)
    return lig_feats
    


# # Make binding residues labels
# for unp in tqdm(asd_entries.keys()):
#     try:
#         lig_feats=load_ligand_info(unp)
#         asd_entries[unp]["binding_residues"]="".join([str(int(x)) for x in lig_feats.max(axis=1)])

#     except Exception as e:
#         asd_entries[unp]["binding_residues"]=""

# json.dump(asd_entries,open(f"{HOME_FOLDER}/ahoj-allosteric/src/method_2/ASD_entries_labels.json","w"),indent=3)

# import sys
# sys.exit(0)

def load_graphs(unp,use_esm=False):
    path=f"{HOME_FOLDER}/allosteric/asd_processing/dgl_graphs/{unp}_graphs.bin"
    if not use_esm:
        try:
            seq=asd_entries[unp]["ref_sequence"]
            seq_feats=torch.tensor(featurize_sequence(seq))
            lig_feats=torch.tensor(load_ligand_info(unp))
        except Exception as e:
            print(e)
            return None
    else:
        embs=load_esm_embs(unp)
        if embs is None:
            return None
        seq_feats=torch.tensor(embs)
    if os.path.exists(path):
        gs=dgl.load_graphs(path)[0]
        final_gs=[]
        for g in gs:
            if not use_esm:
                feats=[]
                if "seq" in ARCHI:
                    feats+=[seq_feats]
                if "geom" in ARCHI:
                    feats+=[g.ndata['feat']]
                if "lig" in ARCHI:
                    feats+=[lig_feats]
                g.ndata['feat']=torch.cat(feats,dim=1)
                pass
            else:
                g.ndata['feat']=torch.cat([g.ndata['feat']],dim=1)
                g.ndata['seq_feat']=seq_feats
            final_gs+=[g]
        return dgl.batch(final_gs)
    else:
        return None

def get_labels(unp):
    try:
        return torch.tensor([float(x) for x in asd_entries[unp]["labels"]]).reshape(-1,1)
    except:
        return None

def get_br(unp):
    try:
        return torch.tensor([float(x) for x in asd_entries[unp]["binding_residues"]]).reshape(-1,1)
    except:
        return None
    
def load_data(fold_id,fold):
    # train_unps=fold["train"]
    # val_unps=fold["val"]
    train_unps=json.load(open(f"{HOME_FOLDER}/ahoj-allosteric/src/method_2/train_unps_fold_{fold_id}.json","r"))
    val_unps=json.load(open(f"{HOME_FOLDER}/ahoj-allosteric/src/method_2/val_unps_fold_{fold_id}.json","r"))
    train_graphs=[(unp,load_graphs(unp)) for unp in tqdm(train_unps)]
    val_graphs=[(unp,load_graphs(unp)) for unp in tqdm(val_unps)]
    train_labels=[(unp,get_labels(unp)) for unp in tqdm(train_unps)]
    val_labels=[(unp,get_labels(unp)) for unp in tqdm(val_unps)]
    train_br=[(unp,get_br(unp)) for unp in tqdm(train_unps)]
    val_br=[(unp,get_br(unp)) for unp in tqdm(val_unps)]

    x_train=[]
    y_train=[]
    y_br_train=[]
    final_train_unps=[]
    for (item1,item2,item3) in zip(train_graphs,train_labels,train_br):
        unp_x,x=item1
        unp_y,y=item2
        unp_z,z=item3
        if x is not None and y is not None and z is not None:
            if y.shape[0]==0:
                continue
            x_train+=[x]
            y_train+=[y]
            y_br_train+=[z]
            final_train_unps+=[unp_x]
    x_val=[]
    y_val=[]
    y_br_val=[]
    final_val_unps=[]
    for (item1,item2,item3) in zip(val_graphs,val_labels,val_br):
        unp_x,x=item1
        unp_y,y=item2
        unp_z,z=item3
        if x is not None and y is not None and z is not None:
            if y.shape[0]==0:
                continue
            x_val+=[x]
            y_val+=[y]
            y_br_val+=[z]
            final_val_unps+=[unp_x]
    # json.dump(final_train_unps,open(f"{HOME_FOLDER}/ahoj-allosteric/src/method_2/train_unps_fold_{fold_id}.json","w"))
    # json.dump(final_val_unps,open(f"{HOME_FOLDER}/ahoj-allosteric/src/method_2/val_unps_fold_{fold_id}.json","w"))
    return x_train,y_train,x_val,y_val,y_br_train,y_br_val

# Training loop
def evaluate(graphs,labels,brs):
    model.eval()
    preds = []
    targets = []
    idx=list(range(len(graphs)))
    batch_size=8
    with torch.no_grad():
        for a in tqdm(range(0,len(idx),batch_size),desc="Evaluating"):
            b=a+batch_size
            xb=[graphs[i].to(device) for i in idx[a:b]]
            yb=torch.cat([labels[i] for i in idx[a:b]],dim=0).reshape(-1,).float().to(device)
            y_br=np.concatenate([brs[i] for i in idx[a:b]],axis=0).flatten()
            # include only BRs
            br_idx=np.argwhere(y_br==1)
            yb=yb[br_idx]
            logits = model(xb).reshape(-1,).float()
            logits=logits[br_idx]
            probs = torch.sigmoid(logits)
            preds.append(probs.cpu().numpy())
            targets.append(yb.cpu().numpy())
            
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    pred_labels = (preds > 0.5).astype(int)
    return {

        "accuracy": accuracy_score(targets, pred_labels),
        "auc": roc_auc_score(targets, preds),
        "aupr": average_precision_score(targets, preds),
        "precision": precision_score(targets, pred_labels),
        "recall": recall_score(targets, pred_labels),
        "f1": f1_score(targets, pred_labels),
        "mcc": matthews_corrcoef(targets, pred_labels)
    }


def print_dataset_stats(graphs,br,labels,mode="train"):
    print(mode,len(graphs),"UNPs")
    br_indices=np.argwhere(np.concatenate(br,axis=0)==1).flatten()
    print(f"{mode} binding residues",len(br_indices))
    y=torch.cat(labels,dim=0)
    y=y[br_indices].numpy()
    n1=(y==1).sum()
    n0=(y==0).sum()
    print(f"{mode} allosteric sites",n1)
    print(f"{mode} non-allosteric sites",n0)
    return n0,n1

if __name__=="__main__":

    FINAL_RESULTS={}
    test_data=asd_dataset["Test"]
    cv_folds=asd_dataset["CV"]
    device="cuda:0"
    for fold_id,fold in enumerate(cv_folds):
        print("Fold",fold_id)
        FINAL_RESULTS[f"FOLD_{fold_id}"]={"train_results":[],
                                          "val_results":[]}
        inp_feat=0
        if "seq" in ARCHI:
            inp_feat+=21 # Onehot representation
        if "geom" in ARCHI:
            inp_feat+=3 # Geometric coordinates offsets
        if "lig" in ARCHI:
            inp_feat+=167 # Ligands MACCs fingerprints

        hidden=128
        output=128
        print("Model architecture",ARCHI)
        print("input feats",inp_feat)
        model=GraphListClassifier(inp_feat,hidden,output).to(device)
        
        # Load data
        train_graphs,train_labels,val_graphs,val_labels,train_br,val_br=load_data(fold_id,fold)
        
        # compute training stats
        n0,n1=print_dataset_stats(train_graphs,train_br,train_labels,mode="train")
        FINAL_RESULTS[f"FOLD_{fold_id}"]["train_stats"]={"unps":len(train_graphs),"positives":int(n0),"negatives":int(n1)}
        
        # Weighted cross entropy
        w1=torch.tensor([n0/n1],dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=w1)

        # compute val stats
        n0,n1=print_dataset_stats(val_graphs,val_br,val_labels,mode="val")
        FINAL_RESULTS[f"FOLD_{fold_id}"]["val_stats"]={"unps":len(val_graphs),"positives":int(n0),"negatives":int(n1)}
        
        epochs = 100
        lr = 1e-4
        optimizer = optim.Adam(model.parameters(), lr=lr)
        batch_size=4 #128
        eval_each=1
        # # Convert to PyTorch
        import random
        train_idx=list(range(len(train_graphs)))
        if TRAIN_MODEL:
            for epoch in tqdm(range(1, epochs + 1),desc="Training"):
                model.train()
                running_loss = 0.0
                total_loss=torch.tensor(0,dtype=torch.float32).to(device)
                for a in (range(0,len(train_idx),batch_size)):
                    b=a+batch_size
                    xb=[train_graphs[i].to(device) for i in train_idx[a:b]]
                    yb=torch.cat([train_labels[i] for i in train_idx[a:b]],dim=0).reshape(-1,).float().to(device)
                    y_br=np.concatenate([train_br[i] for i in train_idx[a:b]],axis=0).flatten()
                    # include only BRs
                    br_idx=np.argwhere(y_br==1).flatten()
                    yb=yb[br_idx]
                    optimizer.zero_grad()
                    logits = model(xb).reshape(-1,).float()
                    # include only BRs
                    logits=logits[br_idx]
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                train_loss = running_loss / len(train_idx)
                random.shuffle(train_idx)
                if epoch%eval_each==0:
                    print(f'Epoch {epoch}')
                    train_eval=evaluate(train_graphs,train_labels,train_br)
                    val_eval=evaluate(val_graphs,val_labels,val_br)
                    FINAL_RESULTS[f"FOLD_{fold_id}"]["train_results"]+=[{x:float(y) for x,y in train_eval.items()}]
                    FINAL_RESULTS[f"FOLD_{fold_id}"]["val_results"]+=[{x:float(y) for x,y in val_eval.items()}]


json.dump(FINAL_RESULTS,open(f"{HOME_FOLDER}/ahoj-allosteric/src/method_2/ASD_gnn_train_results_{ARCHI}.json","w"),indent=3)


