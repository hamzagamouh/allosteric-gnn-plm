import os
from argparse import ArgumentParser

import dgl
import numpy as np
import torch

import dgl
import torch
from scipy.spatial.distance import cdist


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code derived from SE(3)-Transformer (https://github.com/FabianFuchsML/se3-transformer-public/):
# -------------------------------------------------------------------------------------------------------------------------------------

class RandomRotation(object):
    def __init__(self):
        pass

    def __call__(self, x):
        M = np.random.randn(3, 3)
        Q, __ = np.linalg.qr(M)
        return x @ Q


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from GraphTransformer (https://github.com/graphdeeplearning/graphtransformer/):
# -------------------------------------------------------------------------------------------------------------------------------------
def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        """Compute the dot product between source nodes' and destination nodes' representations."""
        return {out_field: (edges.src[src_field] * edges.dst[dst_field])}

    return func


def scaling(field, scale_constant, clip_constant):
    def func(edges):
        """Scale edge representation value using a constant divisor."""
        return {field: ((edges.data[field]) / scale_constant).clamp(-clip_constant, clip_constant)}

    return func


def imp_exp_attn(implicit_attn, explicit_edge):
    """
        implicit_attn: the output of K Q
        explicit_edge: the explicit edge features
    """

    def func(edges):
        """Improve implicit attention scores with explicit edge features, if available."""
        return {implicit_attn: (edges.data[implicit_attn] * edges.data[explicit_edge])}

    return func


def out_edge_features(edge_feat):
    def func(edges):
        """Copy edge features to be passed to FFN_e."""
        return {'e_out': edges.data[edge_feat]}

    return func


def exp(field, clip_constant):
    def func(edges):
        """Clamp edge representations for softmax numerical stability."""
        return {field: torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-clip_constant, clip_constant))}

    return func


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for EGNN-DGL (https://github.com/amorehead/EGNN-DGL):
# -------------------------------------------------------------------------------------------------------------------------------------
def collate(samples):
    graphs, y = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(y)


def glorot_orthogonal(tensor, scale):
    """Initialize a tensor's values according to an orthogonal Glorot initialization scheme."""
    if tensor is not None:
        torch.nn.init.orthogonal_(tensor.data)
        scale /= ((tensor.size(-2) + tensor.size(-1)) * tensor.var())
        tensor.data *= scale.sqrt()


def calculate_and_store_dists_in_graph(graph: dgl.DGLGraph):
    """Derive all node-node distance features from a given batch of DGLGraphs."""
    graphs = dgl.unbatch(graph)
    for graph in graphs:
        graph.edata['c'] = graph.ndata['x'][graph.edges()[0]] - graph.ndata['x'][graph.edges()[1]]
        graph.edata['r'] = torch.sum(graph.edata['c'] ** 2, 1).reshape(-1, 1)
    graph = dgl.batch(graphs)
    return graph


def get_graph(src, dst, pos, node_feature, dtype, undirected=True, num_nodes=None):
    """Construct a single DGLGraph given source and destination node IDs, coordinates, and node and edge features."""
    # src, dst : indices for vertices of source and destination, torch.Tensor
    # pos: x, y, z coordinates of all vertices with respect to the indices, torch.Tensor
    # node_feature: node feature of shape [num_nodes, node_feature_size], torch.Tensor
    # edge_feature: edge feature of shape [num_nodes, edge_feature_size], torch.Tensor
    if num_nodes:
        G = dgl.graph((src, dst), num_nodes=num_nodes)
    else:
        G = dgl.graph((src, dst))
    if undirected:
        G = dgl.to_bidirected(G)
    # Add node features to graph
    G.ndata['f'] = node_feature.type(dtype)
    G.ndata['x'] = pos.type(dtype)  # [num_nodes, 3]
    # Add edge features to graph
    # G.edata['f'] = edge_feature.type(dtype)  # [num_nodes, edge_feature_size]
    return G


def get_rgraph(num_nodes: int, num_edges: int, node_feature_size: int,
               edge_feature_size: int, self_loops: bool, dtype: torch.Type, test: bool):
    G = dgl.rand_graph(num_nodes, num_edges)
    if not self_loops:
        G = dgl.remove_self_loop(G)  # Keep self-loops out of each randomly-generated graph for compatibility with EGNNs
        graph_edges = torch.stack(G.edges(), dim=0).T
        for edge_i in range(num_nodes):
            for edge_j in range(num_nodes):
                if edge_i != edge_j:
                    random_edge = torch.tensor([edge_i, edge_j])
                    should_add_edge = True
                    for edge in graph_edges:
                        # Ascertain whether the randomly-generated edge already exists
                        should_add_edge = not torch.equal(random_edge, edge)
                        if not should_add_edge:
                            break
                    if should_add_edge:
                        G = dgl.add_edges(G, random_edge[0], random_edge[1])
                        graph_edges = torch.stack(G.edges(), dim=0).T
                        edge_count_is_full = num_edges - len(G.edges()[0]) == 0
                        if edge_count_is_full:
                            break
    if test:
        src = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        dst = torch.tensor([1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2])
    else:
        src = G.edges()[0]
        dst = G.edges()[1]
    # Add node features to graph
    pos = torch.ones(num_nodes, 3) if test else torch.rand((num_nodes, 3))  # [num_nodes, 3]
    node_features = torch.ones(num_nodes, node_feature_size) if test else torch.rand((num_nodes, node_feature_size))
    # Add edge features to graph
    edge_features = torch.ones(num_edges, edge_feature_size) if test else torch.rand((num_edges, edge_feature_size))
    return get_graph(src, dst, pos, node_features, edge_features, dtype, False, num_nodes=num_nodes)


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code derived from egnn-pytorch (https://github.com/lucidrains/egnn-pytorch/blob/main/egnn_pytorch/utils.py):
# -------------------------------------------------------------------------------------------------------------------------------------
def rot_z(gamma):
    return torch.tensor([
        [torch.cos(gamma), -torch.sin(gamma), 0],
        [torch.sin(gamma), torch.cos(gamma), 0],
        [0, 0, 1]
    ], dtype=gamma.dtype)


def rot_y(beta):
    return torch.tensor([
        [torch.cos(beta), 0, torch.sin(beta)],
        [0, 1, 0],
        [-torch.sin(beta), 0, torch.cos(beta)]
    ], dtype=beta.dtype)


def rotate(alpha, beta, gamma):
    return rot_z(alpha) @ rot_y(beta) @ rot_z(gamma)

def featurize_biopython_atom(atom):
    """
    Featurize a Biopython Atom object based on its element.
    Returns a feature vector (list of numerical values).
    """

    # Define a small one-hot encoding for common elements
    element = atom.element.upper()
    common_elements = ['C', 'N', 'O', 'S', 'H', 'P', 'FE', 'MG', 'ZN', 'CA']
    one_hot = [1 if element == e else 0 for e in common_elements]

    # If it's not in the list, use 'unknown' category
    if element not in common_elements:
        one_hot.append(1)  # unknown element
    else:
        one_hot.append(0)

    # Optional: Add physicochemical properties (very basic)
    # Atomic number (fallback: 0)
    atomic_number = {
        'H': 1, 'C': 6, 'N': 7, 'O': 8, 'S': 16, 'P': 15,
        'FE': 26, 'MG': 12, 'ZN': 30, 'CA': 20
    }.get(element, 0)

    # Final feature vector
    features = one_hot + [atomic_number]
    return features


def featurize_biopython_residue(residue):
    """
    Featurize a Biopython Residue object into a numerical vector.
    Returns a feature vector (list of numerical values).
    """

    # List of 20 standard amino acids (3-letter codes)
    standard_residues = [
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY',
        'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
        'THR', 'TRP', 'TYR', 'VAL'
    ]

    resname = residue.get_resname().upper()
    
    # One-hot encoding for residue name
    one_hot = [1 if resname == aa else 0 for aa in standard_residues]
    
    # Unknown residue flag
    one_hot.append(1 if resname not in standard_residues else 0)

    # Optional numeric features
    features = one_hot.copy()

    # Average B-factor across atoms in the residue (can be useful)
    try:
        b_factors = [atom.get_bfactor() for atom in residue.get_atoms()]
        avg_b = sum(b_factors) / len(b_factors) if b_factors else 0.0
    except:
        avg_b = 0.0

    features.append(avg_b)

    return features


def featurize_atom_residue(atom,residue):
    return featurize_biopython_atom(atom)+featurize_biopython_residue(residue)


from Bio.PDB import Superimposer
from Bio.SVDSuperimposer import SVDSuperimposer


def get_atoms_and_features(residues,ref_residues,state):    
    atoms = []
    atom_feats = []
    coords = []
    ca_atoms=[]
    # ref_ca_atoms=[]
    # for residue in residues:
    #     if 'CA' in residue:
    #         ca_atoms.append(residue['CA'])
    # for residue in ref_residues:
    #     if 'CA' in residue:
    #         ref_ca_atoms.append(residue['CA'])

    # if len(residues)==len(ref_residues):
    #     super_imposer = Superimposer()
    #     super_imposer.set_atoms(ref_ca_atoms,ca_atoms)
    #     print("RMSD",super_imposer.rms)
    # else:
    super_imposer=None

    for residue in residues:
        for atom in residue:
            if super_imposer:
                super_imposer.apply([atom])
            atoms.append(atom)
            atom_feats.append(featurize_atom_residue(atom,residue)+list(atom.coord)+[state])
            coords.append(atom.coord)
                    
    coords = np.array(coords, dtype=np.float32)
    return atoms, atom_feats, coords



def build_protein_graph(atom_features, coords, cutoff=6.0):
    # Cutoff --> 2 - 5 
    num_atoms = coords.shape[0]
    
    # Create edges based on distance threshold
    dist_matrix = cdist(coords, coords)
    src, dst = np.where((dist_matrix < cutoff) & (dist_matrix >= 0))  # avoid self-loops
    
    # Build graph
    g = dgl.graph((src, dst), num_nodes=num_atoms)
    
    # Set node features
    g.ndata['f'] = torch.tensor(atom_features)
    g.ndata['x'] = torch.tensor(coords)
    
    return g

from Bio.PDB.Polypeptide import one_to_three,three_to_one,is_aa


def get_aa_name(x):
    try:
        return one_to_three(x)
    except:
        return "X"


def featurize_sequence(sequence):
    """
    Featurize a Biopython Residue object into a numerical vector.
    Returns a feature vector (list of numerical values).
    """

    # List of 20 standard amino acids (3-letter codes)
    standard_residues = [
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY',
        'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
        'THR', 'TRP', 'TYR', 'VAL'
    ]

    feats=[]
    for x in sequence:
        resname=get_aa_name(x)
        # One-hot encoding for residue name
        one_hot = [1 if resname == aa else 0 for aa in standard_residues]
        
        # Unknown residue flag
        one_hot.append(1 if resname not in standard_residues else 0)
        feats+=[one_hot]
        


    return np.array(feats)