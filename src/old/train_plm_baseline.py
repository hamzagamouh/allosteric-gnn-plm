import sys
sys.path.append("/storage/brno12-cerit/home/hamzagamouh/.local/lib/python3.8/site-packages")

import json

import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import os
from e_gnn_utils import get_atoms_and_features,build_protein_graph,featurize_sequence
HOME_FOLDER="/storage/praha1/home/hamzagamouh"

pocket_mapping=json.load(open(f"{HOME_FOLDER}/allosteric/ahoj_processing/pocket_mapping.json",'r'))

ahoj_queries=json.load(open(f"{HOME_FOLDER}/allosteric/ahoj_processing/non_catalytic_csa_pockets_info.json",'r'))

new_df=pd.read_csv(f"{HOME_FOLDER}/allosteric/ahoj_processing/casbench_processed_csa_pockets.csv")

print(len(new_df[new_df["NON_CATALYTIC_CSA_POCKET"]!="N/A"]["Entry"].unique()),"casbench entries")


non_allo_casbench_pockets=new_df[new_df["LABEL"]=="Non_Allosteric"]["NON_CATALYTIC_CSA_POCKET"].unique().tolist()
allo_casbench_pockets=new_df[new_df["LABEL"]=="Allosteric"]["NON_CATALYTIC_CSA_POCKET"].unique().tolist()

casbench_pockets=[x for x in non_allo_casbench_pockets+allo_casbench_pockets if isinstance(x,str)]

df=pd.read_csv(f"{HOME_FOLDER}/allosteric/ahoj_processing/casbench_processed_csa_pockets.csv")

valid_pockets=[x.replace(".pkl","") for x in os.listdir(f"{HOME_FOLDER}/allosteric/method_2/") if ".pkl" in x]

new_df=df[df["NON_CATALYTIC_CSA_POCKET"].isin(valid_pockets)]
allosteric_pockets=df[df["LABEL"]=="Allosteric"]["NON_CATALYTIC_CSA_POCKET"].unique()

dataset=json.load(open(f"{HOME_FOLDER}/allosteric/sequence_dataset.json",'r'))
embs_path=f"{HOME_FOLDER}/allosteric/method_1/non_catalytic_csa"


dpocket_df=pd.read_csv(f"{HOME_FOLDER}/allosteric/method_2/dpocket_feats/all_outputs.txt")
cols=[x for x in dpocket_df.columns if x not in ["pdb","lig"]]

def normalize_entry(x):
    return "-".join(x.split("_"))

all_dpocket_feats={normalize_entry(x):y for x,y in zip(dpocket_df.loc[:,"pdb"],dpocket_df.loc[:,cols].values)}

def compute_stats(arr):
    # Compute statistics along axis=0 (i.e., per feature)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    min_ = np.min(arr, axis=0)
    max_ = np.max(arr, axis=0)
    median = np.median(arr, axis=0)
    stats = np.stack([mean,std,min_,max_,median], axis=0).flatten()
    return stats

from sklearn.decomposition import PCA

def reduce_pca(X, n_components=2):
    """
    Reduces the dimensionality of matrix X using PCA.

    Parameters:
        X (array-like or DataFrame): The input data matrix (samples x features).
        n_components (int or float): Number of components to keep. Can be:
            - int: number of components
            - float between 0 and 1: percentage of variance to keep

    Returns:
        X_reduced (ndarray): The PCA-reduced data matrix.
        pca (PCA object): The fitted PCA object (can be used for inverse_transform, etc.)
    """
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    return X_reduced, pca

def load_data(fold,mode,train_pca=None):
    xs=[]
    ys=[]

    dataset_entries=json.load(open(f"{HOME_FOLDER}/allosteric/method_1/casbench_enriched_dataset/{mode}_{fold}.json",'r'))
    dataset_entries=[x for x in dataset_entries if x in  casbench_pockets]   

    esm_feats=[]
    for casbench_pocket in tqdm(dataset_entries):
        try:
            seq_arr=featurize_sequence(dataset[casbench_pocket][0]["sequence"])
            if seq_arr.shape[0]==0 :
                continue
            esm_arr=np.load(f"{embs_path}/{casbench_pocket}_esm_embs.npy")
            labels=np.array([int(x) for x in dataset[casbench_pocket][0]["allosteric_residues"]])
            idx=np.argwhere(labels==1).flatten()
            pocket_structures=pickle.load(open(f"{HOME_FOLDER}/allosteric/method_2/{casbench_pocket}_structures.pkl",'rb'))
            pocket_ligands=pickle.load(open(f"{HOME_FOLDER}/allosteric/method_2/{casbench_pocket}_ligands.pkl",'rb'))
            lig_feats=np.array(pocket_ligands).max(axis=0).reshape(1,-1)
            # all_feats=np.concatenate([all_dpocket_feats[x].reshape(1,-1) for x in pocket_structures],axis=0)
            # dpocket_feats=compute_stats(all_feats)
            if esm_arr.shape[0]==0:
                continue
            # xs+=[dpocket_feats.reshape(1,-1)]
            # xs+=[np.concatenate([esm_arr[idx].mean(axis=0).reshape(1,-1)],axis=1)]
            # ,dpocket_feats.reshape(1,-1)
            # xs+=[np.concatenate([esm_arr[idx].mean(axis=0).reshape(1,-1),lig_feats],axis=1)]
            # xs+=[np.concatenate([seq_arr[idx].max(axis=0).reshape(1,-1)],axis=1)]
            # xs+=[np.concatenate([esm_arr[idx].mean(axis=0).reshape(1,-1)],axis=1)]
            xs+=[lig_feats]
            # xs+=[np.concatenate([seq_arr[idx].max(axis=0).reshape(1,-1),dpocket_feats.reshape(1,-1)],axis=1)]
            # xs+=[np.concatenate([seq_arr[idx].max(axis=0).reshape(1,-1)],axis=1)]

            # xs+=[]
            if casbench_pocket in allosteric_pockets:
                ys+=[1]
            else:
                ys+=[0]
        except:
            continue
    
    # xs=np.concatenate(xs,axis=0)
    # esm_feats=np.concatenate(esm_feats,axis=0)
    # if mode=="train":
    #     esm_feats,train_pca=reduce_pca(esm_feats)
    # else:
    #     esm_feats=train_pca.transform(esm_feats)
    return np.concatenate(xs,axis=0),np.array(ys),None
    # return np.concatenate([esm_feats,xs],axis=1),np.array(ys),train_pca
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, matthews_corrcoef
import numpy as np


def evaluate(y_true,y_pred_proba):
    # AUC using predicted probabilities
    auc = roc_auc_score(y_true, y_pred_proba)

    # MCC using predicted class labels
    y_pred = (y_pred_proba >= 0.5).astype(int)
    mcc = matthews_corrcoef(y_true, y_pred)
    return auc,mcc

# To store results
# models = {
#         "Logistic (no reg)": LogisticRegression(penalty='none', solver='saga', max_iter=10000),
#         "Logistic (L1 1000)": LogisticRegression(penalty='l1', C=1000, solver='liblinear', max_iter=10000),
#         "Logistic (L1 100)": LogisticRegression(penalty='l1', C=100, solver='liblinear', max_iter=10000),
#         "Logistic (L1 10)": LogisticRegression(penalty='l1', C=10, solver='liblinear', max_iter=10000),
#         "Logistic (L1 1)": LogisticRegression(penalty='l1', C=1, solver='liblinear', max_iter=10000),
#         "Logistic (L1 0.1)": LogisticRegression(penalty='l1', C=0.1, solver='liblinear', max_iter=10000),
#         "Logistic (L1 0.01)": LogisticRegression(penalty='l1', C=0.01, solver='liblinear', max_iter=10000),
#         "Logistic (L1 0.001)": LogisticRegression(penalty='l1', C=0.001, solver='liblinear', max_iter=10000)
#     }
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Create a dictionary of RandomForestClassifier models with different max_features
# models = {
#     f"rf_maxfeat_{round(mf, 1)}": RandomForestClassifier(max_features=mf, n_estimators=100)
#     for mf in np.arange(0.1, 1.1, 0.1)  # from 0.1 to 1.0 inclusive
# }

models={
        # "Logistic":LogisticRegression(penalty='none', solver='saga', max_iter=10000),
        "RF":RandomForestClassifier(n_estimators=100),
        # "SVC Linear":SVC(kernel='linear', C=1.0, probability=True),
        # "SVC RBF":SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
        }
# models={f"Logistic (L1 {c})": LogisticRegression(penalty='l1', C=c, solver='liblinear', max_iter=10000) for c in [0.001,0.01,0.1,1,10,100,1000]}

# models["Logistic (no reg)"]= LogisticRegression(penalty='none', solver='saga', max_iter=10000)

results = {x:{"AUC": [], "MCC": []} for x in models.keys()}



TRAIN_MODEL=True

if TRAIN_MODEL:

    for fold in range(10):
        # x_train,y_train,train_pca=load_data(fold,mode="train")
        # x_val,y_val,_=load_data(fold,mode="val",train_pca=train_pca)
        x_train,y_train,train_pca=load_data(fold,mode="train")
        x_val,y_val,_=load_data(fold,mode="val",train_pca=train_pca)

        print("train",x_train.shape,y_train.shape)
        print("val",x_val.shape,x_val.shape)
        for name, model in models.items():
            model.fit(x_train, y_train)
            y_pred_proba = model.predict_proba(x_val)[:, 1]
            y_train_pred_proba = model.predict_proba(x_train)[:, 1]

            auc_train,mcc_train=evaluate(y_train,y_train_pred_proba)
            auc_val,mcc_val=evaluate(y_val,y_pred_proba)
            

            results[name]["AUC"].append([auc_train,auc_val])
            results[name]["MCC"].append([mcc_train,mcc_val])




    for model in results:
        auc_scores=np.array(results[model]["AUC"])
        train_auc_scores=auc_scores[:,0]
        val_auc_scores = auc_scores[:,1]

        mcc_scores=np.array(results[model]["MCC"])
        train_mcc_scores=mcc_scores[:,0]
        val_mcc_scores = mcc_scores[:,1]
        print("Val AUC",val_auc_scores)
        print(f"{model} - Mean Train AUC: {np.mean(train_auc_scores):.4f}, Mean Train MCC: {np.mean(train_mcc_scores):.4f}")
        print(f"{model} - Mean Val AUC: {np.mean(val_auc_scores):.4f}, Mean Val MCC: {np.mean(val_mcc_scores):.4f}")


else:
    # model=RandomForestClassifier(n_estimators=100)
    # model=RandomForestClassifier(n_estimators=100)
    model=LogisticRegression(penalty='none', solver='saga', max_iter=10000)
    # Final training and evaluation
    x_train,y_train,_=load_data(fold=0,mode="train")
    x_val,y_val,_=load_data(fold=0,mode="val")
    X_dev=np.concatenate([x_train,x_val],axis=0)
    y_dev=np.concatenate([y_train,y_val],axis=0)
    x_test,y_test,_=load_data(fold="final",mode="test")

    print(X_dev.shape)

    model.fit(X_dev, y_dev)
    y_pred_proba = model.predict_proba(x_test)[:, 1]
    auc_val,mcc_val=evaluate(y_test,y_pred_proba)

    print("Test AUC",auc_val)
    print("Test MCC",mcc_val)

# Merge dpocket outputs

# def parse_file(file,entry):
#     count=0
#     with open(file,'r') as f:
#         lines=[line.strip() for line in f]
    
#     cols=",".join(lines[0].split())
#     row=lines[1].split()
#     row[0]=entry
#     row=",".join(row)
#     return cols,row




# with open(f"{HOME_FOLDER}/allosteric/method_2/dpocket_feats/all_outputs.txt","w") as f:
#     for k,entry in enumerate(tqdm(os.listdir(f"{HOME_FOLDER}/allosteric/method_2/dpocket_feats/outputs"))):
#         try:
#             file=f"{HOME_FOLDER}/allosteric/method_2/dpocket_feats/outputs/{entry}/{entry}.txt"
#             cols,row=parse_file(file,entry)
#             if k==0:
#                 f.write(f"{cols}\n")
#             f.write(f"{row}\n")
#         except:
#             print('Problem in entry',entry)