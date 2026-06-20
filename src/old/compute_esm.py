print("Exporting libraries...")
import sys
sys.path.append("/storage/brno12-cerit/home/hamzagamouh/.local/lib/python3.8/site-packages")


import torch
import esm
import numpy as np
import os
from Bio import SeqIO
import gc
THRESHOLD = 1022
# THRESHOLD = 800

# Load ESM-2 model
# model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
print("Loading ESM-2 model ...")
# model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
# print("Loading ESM-2 model ...")
model,alphabet = esm.pretrained.load_model_and_alphabet_local("/storage/praha1/home/hamzagamouh/.cache/torch/hub/checkpoints/esm2_t36_3B_UR50D.pt")
print(model,alphabet)
# model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
output_dir = 'embeddings-3B-for-Matyas'
batch_converter = alphabet.get_batch_converter()
device = torch.device(f"cuda:0" if (torch.cuda.is_available()) else "cpu")
device="cpu"
model.to(device)

print(device)

def get_esm_emb(sequence):
    name=""
    vectors = []
    print(len(sequence),"residues")
    while len(sequence) > 0:
        sequence1 = sequence[:THRESHOLD]
        sequence = sequence[THRESHOLD:]
        data = [
            (name, sequence1)
        ]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        # if device=="cuda:1":
        batch_tokens = batch_tokens.to(device)
        # Extract per-residue representations (on GPU)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]
        vectors1 = token_representations.detach().cpu().numpy()[0][1:-1]
        if len(vectors) > 0:
            vectors = np.concatenate((vectors, vectors1))
        else:
            vectors = vectors1

        del results, token_representations, batch_tokens
        # if device=="cuda:0":
        torch.cuda.empty_cache()   
        gc.collect()

    return np.array(vectors)

from fileinput import filename
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Bio import SeqIO
import pickle
import shutil
import zipfile

# assert torch.cuda.is_available(), "CUDA is not available"

import warnings
warnings.filterwarnings("ignore")


# EMBEDDER=get_embedder("t5")


# def compute_t5_embs(seq):
#     return EMBEDDER.embed(seq)


import json
from tqdm import tqdm

HOME_FOLDER="/storage/praha1/home/hamzagamouh"

asd_entries=json.load(open(f"{HOME_FOLDER}/ahoj-allosteric/src/method_2/ASD_entries.json","r"))

for unp,data in tqdm(asd_entries.items()):
    if os.path.exists(f"{HOME_FOLDER}/allosteric/method_2/ASD_{unp}_esm_embs.npy"):
        continue
    try:
        sequence=data["ref_sequence"]
        embs=get_esm_emb(sequence)
        print(embs.shape)
        np.save(f"{HOME_FOLDER}/allosteric/method_2/ASD_{unp}_esm_embs.npy",embs)
    except Exception as e:
        print("Error on entry",unp)
        print(e)
        torch.cuda.empty_cache()   
        gc.collect()
        continue