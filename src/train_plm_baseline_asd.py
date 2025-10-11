import sys
sys.path.append("/storage/brno12-cerit/home/hamzagamouh/.local/lib/python3.8/site-packages")

import json
from tqdm import tqdm
import numpy as np
import os
HOME_FOLDER="/storage/praha1/home/hamzagamouh"

asd_dataset=json.load(open(f"{HOME_FOLDER}/allosteric/method_2/ASD_dataset.json","r"))
asd_entries=json.load(open(f"{HOME_FOLDER}/ahoj-allosteric/src/method_2/ASD_entries.json","r"))


def load_esm_embs(unp):
    path=f"{HOME_FOLDER}/allosteric/method_2/ASD_{unp}_esm_embs.npy"
    if os.path.exists(path):
        return np.load(path)
    else:
        return None
    
def load_labels(unp):
    try:
        labels=np.array([int(x) for x in asd_entries[unp]["labels"]])
        if len(labels)>0:
            return labels
        return None
    except Exception as e:
        return None
    

def get_br(unp):
    try:
        return torch.tensor([float(x) for x in asd_entries[unp]["binding_residues"]]).reshape(-1,1)
    except:
        return None

# def get_data(fold):
#     train_unps=fold["train"]
#     val_unps=fold["val"]
def get_data(train_unps,val_unps):
    X_train=[]
    y_train=[]
    y_train_br=[]
    for unp in tqdm(train_unps):
        embs=load_esm_embs(unp)
        labels=load_labels(unp)
        br=get_br(unp)
        if embs is not None and labels is not None and br is not None:
            X_train+=[embs]
            y_train+=[labels]
            y_train_br+=[br]
    X_train=np.concatenate(X_train,axis=0)
    y_train=np.concatenate(y_train,axis=0)
    y_train_br=np.concatenate(y_train_br,axis=0)

    X_val=[]
    y_val=[]
    y_val_br=[]
    for unp in tqdm(val_unps):
        embs=load_esm_embs(unp)
        labels=load_labels(unp)
        br=get_br(unp)
        if embs is not None and labels is not None and br is not None:
            X_val+=[embs]
            y_val+=[labels]
            y_val_br+=[br]
    X_val=np.concatenate(X_val,axis=0)
    y_val=np.concatenate(y_val,axis=0)
    y_val_br=np.concatenate(y_val_br,axis=0)
    return X_train,y_train,y_train_br,X_val,y_val,y_val_br

test_data=asd_dataset["Test"]
cv_folds=asd_dataset["CV"]
    


from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    precision_score, recall_score, matthews_corrcoef,f1_score
)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a PyTorch MLP model class
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, activation=nn.ReLU(),dropout=True):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation)
            if dropout:
                layers.append(nn.Dropout(0.1))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))  # binary classification output
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)  # (batch,)

def evaluate(targets,pred_labels,preds):
    eval={
        "accuracy": accuracy_score(targets, pred_labels),
        "auc": roc_auc_score(targets, preds),
        "aupr": average_precision_score(targets, preds),
        "precision": precision_score(targets, pred_labels),
        "recall": recall_score(targets, pred_labels),
        "f1": f1_score(targets, pred_labels),
        "mcc": matthews_corrcoef(targets, pred_labels)
    }
    return eval



def train_mlp(x_train, y_train,y_train_br, x_val, y_val,y_val_br, hidden_layers, activation, epochs=50, lr=0.001):
    input_dim = x_train.shape[1]
    model = MLP(input_dim, hidden_layers, activation).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_br_idx=np.argwhere(y_train_br==1)
    y_train=y_train[train_br_idx]
    val_br_idx=np.argwhere(y_val_br==1)

    y_val=y_val[val_br_idx]
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device).reshape(-1)
    n0=(y_train_tensor == 0).sum()
    n1=(y_train_tensor == 1).sum()
    if n1>0 and n0>0:
        w1=n0/n1
    else:
        w1=torch.tensor([1],dtype=torch.float32).to(device)
    print(w1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=w1)

    x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)

    model.train()
    train_evals=[]
    val_evals=[]
    for epoch in (range(epochs)):
        print("Epoch",epoch)
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train_tensor).reshape(-1,)[train_br_idx]
        loss = criterion(outputs.reshape(-1), y_train_tensor.reshape(-1))
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            train_logits = model(x_train_tensor).cpu().numpy().reshape(-1,)[train_br_idx]
            train_probs = 1 / (1 + np.exp(-train_logits))  # sigmoid
            train_preds = (train_probs >= 0.5).astype(int)
            val_logits = model(x_val_tensor).cpu().numpy().reshape(-1,)[val_br_idx]
            val_probs = 1 / (1 + np.exp(-val_logits))  # sigmoid
            val_preds = (val_probs >= 0.5).astype(int)

        y_train=y_train.reshape(-1)
        train_preds=train_preds.reshape(-1)
        train_probs=train_probs.reshape(-1)
        y_val=y_val.reshape(-1)
        val_preds=val_preds.reshape(-1)
        val_probs=val_probs.reshape(-1)

        train_evals+=[evaluate(y_train,train_preds,train_probs)]
        val_evals+=[evaluate(y_val,val_preds,val_probs)]
        
    return train_evals,val_evals

results = []
EPOCHS=100

# Loop through folds

mlp_settings = [
    {'hidden_layers': [1024]*3, 'activation': nn.ReLU()},
    # {'hidden_layers': [128, 64], 'activation': nn.ReLU()},
    # {'hidden_layers': [32, 32, 32], 'activation': nn.ReLU()}
]

for fold_id,fold in enumerate(cv_folds):

    train_unps=json.load(open(f"{HOME_FOLDER}/ahoj-allosteric/src/method_2/train_unps_fold_{fold_id}.json","r"))
    val_unps=json.load(open(f"{HOME_FOLDER}/ahoj-allosteric/src/method_2/val_unps_fold_{fold_id}.json","r"))
    x_train,y_train,y_train_br,x_val,y_val,y_val_br=get_data(train_unps,val_unps)
    print(x_train.shape,y_train.shape,'BR',(y_train_br==1).sum())
    print(x_val.shape,y_val.shape,'BR',(y_val_br==1).sum())

    for params in mlp_settings:
        train_evals,val_evals = train_mlp(x_train, y_train,y_train_br, x_val, y_val,y_val_br,
                                 hidden_layers=params['hidden_layers'],
                                 activation=params['activation'],
                                 epochs=EPOCHS, lr=0.001)

        results.append({
            'Model': 'MLP (PyTorch)',
            'Fold': fold_id,
            'Train results': train_evals,
            'Val results': val_evals,
        })


json.dump(results,open(f"{HOME_FOLDER}/ahoj-allosteric/src/method_2/ASD_train_results_PLM.json","w"),indent=3)
