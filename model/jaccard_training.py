import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import sys
import os
from scipy.spatial.distance import jaccard, cosine
torch.cuda.empty_cache()
from xgboost import XGBClassifier, DMatrix

from model import DTIModel
from barlowdti_xxl import get_ecfps, get_emb


dt_str = "14062024_0910"

model_xxl = DTIModel(
    bt_model_path=f"../model/stash/{dt_str}",
    gbm_model_path=f"../model/{dt_str}_barlowdti_xxl_model.json",
)

train = pd.read_pickle("../dataset/big_data_undersampled.pkl").drop_duplicates()

seq = "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"
smiles = [
    "CC(C)C[N@](C[C@H]([C@H](Cc1ccccc1)NC(=O)O[C@H]2CCOC2)O)S(=O)(=O)c3ccc(cc3)N"
]

sel_smiles_str = "smiles"
sel_seq_str = "1HPV"

leaf_indices = model_xxl.predict(smiles, seq, pred_leaf=True)[0]
leaf_indices = np.array(leaf_indices)

# Check if train_leaf_indices is already computed
if os.path.exists("../proteomics/train_leaf_indices_big_data_undersampled.npy"):
    train_leaf_indices = np.load("../proteomics/train_leaf_indices_big_data_undersampled.npy", allow_pickle=True)
    # np.save("../proteomics/train_leaf_indices_big_data_undersampled.npy", train_leaf_indices, allow_pickle=True)
else:
    train_ecfp = get_ecfps(train)
    print("Encoded ECFP", train_ecfp.shape)
    train_emb = get_emb(train)
    print("Encoded Emb", train_emb.shape)
    train_bt = model_xxl.bt_model.zero_shot(train_ecfp, train_emb)
    print("Encoded BT", train_bt.shape)
    d_train_bt = DMatrix(train_bt)

    train_leaf_indices = model_xxl.gbm_model.get_booster().predict(d_train_bt, pred_leaf=True)
    train_leaf_indices = np.array(train_leaf_indices)
    print("Got Train Leaf Indices", train_leaf_indices.shape)

    np.save("../proteomics/train_leaf_indices_big_data_undersampled.npy", train_leaf_indices, allow_pickle=True)

jaccard_dis = np.zeros(len(train_leaf_indices))

for i, train_leaf in tqdm(enumerate(train_leaf_indices)):
    jaccard_dis[i] = jaccard(leaf_indices, train_leaf)

jaccard_sim = 1 - jaccard_dis

train["jaccard_sim"] = jaccard_sim
train.sort_values("jaccard_sim", ascending=False, inplace=True)

train.head(100).to_csv(f"../proteomics/top100influential_{sel_seq_str}.csv", index=False)
train.to_pickle(f"../proteomics/train_with_jaccard_{sel_seq_str}.pkl")
