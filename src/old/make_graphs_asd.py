print("Exporting library..")
import sys
sys.path.append("/storage/brno12-cerit/home/hamzagamouh/.local/lib/python3.8/site-packages")

import torch
import os
import pickle
import numpy as np
import dgl

import json,requests
from tqdm import tqdm
HOME_FOLDER="/storage/praha1/home/hamzagamouh"
asd_entries=json.load(open(f"{HOME_FOLDER}/ahoj-allosteric/src/method_2/ASD_entries.json","r"))

import numpy as np
from scipy.spatial.distance import cdist

def adjacency_matrix_with_threshold(points: np.ndarray, threshold: float = 8.0) -> np.ndarray:
    """
    Compute an adjacency matrix for a set of 3D points based on a Euclidean distance threshold.
    
    Parameters:
        points (np.ndarray): An (N, 3) array of N points in 3D.
        threshold (float): The distance threshold for creating edges.
    
    Returns:
        np.ndarray: An (N, N) adjacency matrix where entry (i, j) is 1 if distance < threshold, else 0.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Input must be an array of shape (N, 3)")

    distances = cdist(points, points, metric='euclidean')
    adjacency = (distances < threshold).astype(int)

    return adjacency


def create_dgl_graph(A,node_feats):
    """
    Create a DGL graph from an adjacency matrix A and add node features.

    Parameters:
    A (np.ndarray or torch.Tensor): Adjacency matrix of shape (N, N)

    Returns:
    DGLGraph: A DGL graph with node features added
    """
    # Ensure the adjacency matrix is a torch tensor
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float32)

    # Get edge indices from the adjacency matrix (non-zero entries)
    src, dst = torch.nonzero(A, as_tuple=True)

    # Create the graph
    g = dgl.graph((src, dst), num_nodes=A.shape[0])

    assert node_feats.shape[0]==A.shape[0]
    # Add node features
    g.ndata['feat'] = node_feats

    return g

def compute_offsets(residues,ref_residues):
    n=len(residues)
    offsets=np.zeros((n,3))
    for k,(x,x_ref) in enumerate(zip(residues,ref_residues)):
        if x is not None and x_ref is not None:
            offsets[k]=x_ref.coord-x.coord
    return offsets

def make_graph(residues,ref_residues):
    idx=[]
    coords=[]
    for k,x in enumerate(residues):
        if x is not None:
            idx+=[k]
            coords+=[x.coord]
    coords=np.array(coords)
    adj=adjacency_matrix_with_threshold(coords)
    n=len(residues)
    A=np.zeros((n,n))
    for i in range(len(idx)):
        A[idx[i],idx]=adj[i,:]
    
    # Offsets from the superimposition
    offsets=compute_offsets(residues,ref_residues)
    node_feats=np.concatenate([offsets],axis=1)
    g=create_dgl_graph(A,torch.tensor(node_feats, dtype=torch.float32))

    return g


def process_entry(unp):
    try:
        if os.path.exists(f"{HOME_FOLDER}/allosteric/asd_processing/{unp}_residues.pkl"):
            if os.path.exists(f"{HOME_FOLDER}/allosteric/asd_processing/dgl_graphs/{unp}_graphs.bin"):
                return
            new_info=pickle.load(open(f"{HOME_FOLDER}/allosteric/asd_processing/{unp}_residues.pkl",'rb'))
            ref_info=pickle.load(open(f"{HOME_FOLDER}/allosteric/asd_processing/{unp}_ref_residues.pkl",'rb'))
            all_graphs=[]
            for k,v in ref_info.items():
                if "modulator" not in k:
                    ref_residues=v
            for k,residues in (new_info.items()):
                if "modulator" in k:
                    continue
                all_graphs+=[make_graph(residues,ref_residues)]
        
            dgl.save_graphs(f"{HOME_FOLDER}/allosteric/asd_processing/dgl_graphs/{unp}_graphs.bin", all_graphs)
    except Exception as e:
        print(f"Error in {unp} ---> {e}")

from joblib import Parallel,delayed
Parallel(n_jobs=4, backend='multiprocessing', verbose=1)(
    delayed(process_entry)(unp) for unp in asd_entries.keys()
)

# To load graphs
# unp="P0A7Z4"
# loaded_graphs, _ = dgl.load_graphs(f"{HOME_FOLDER}/allosteric/asd_processing/dgl_graphs/{unp}_graphs.bin")
