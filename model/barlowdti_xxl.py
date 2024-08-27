import numpy as np
import pandas as pd
import pickle
import joblib
import sys
import os
from tqdm.auto import tqdm, trange
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from gbm import XGBCls, XGBClsOptuna
from barlow_twins import BarlowTwins

# Add the path to custom modules if necessary
sys.path.append("../utils/")

# Assuming sequence.py contains the encode_sequences function
from sequence import encode_sequences


def get_ecfps(df, smiles_col="smiles"):
    """Generate ECFPs from SMILES in the given DataFrame."""
    mols = [Chem.MolFromSmiles(smi) for smi in df[smiles_col]]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in mols]
    return np.array(fps)


def get_emb(df, protein_col="sequence", model="prost_t5"):
    """Generate embeddings from target sequences in the given DataFrame."""
    emb = encode_sequences(df[protein_col].tolist(), model)
    return np.array([np.array(x) for x in emb])


def load_dataset():
    """Load train sets for a given dataset."""
    base_path = f"../dataset/"
    train = pd.read_pickle(base_path + "barlow_xxl_data_undersampled.pkl").drop_duplicates()
    # train = pd.read_csv(base_path + "all_train_val.csv")
    train_ecfp, train_emb = get_ecfps(train), get_emb(train)
    train_labels = train["label"].values

    print(f"Balance ratio: {np.mean(train_labels)}")

    return train_ecfp, train_emb, train_labels


def evaluate_model(
        model, 
        train_features=None,
        valid_features=None,
        test_features=None,
        train_labels=None,
        valid_labels=None,
        test_labels=None
):
    """Train a model and evaluate it on the test set."""
    try:
        model.optimize(train_features, train_labels, valid_features, valid_labels)
    except AttributeError:
        print("Model does not support optimization. Skipping optimization.")

    model.fit(train_features, train_labels)
    preds = model.predict_proba(test_features)[:, 1]
    roc = roc_auc_score(test_labels, preds)
    pr = average_precision_score(test_labels, preds)
    return {"ROC": roc, "PR": pr}


def train(precomputed_features, bt_model):
    """Benchmark performance across models for a given dataset, across different seeds."""
    train_ecfp, train_emb, train_y = precomputed_features
    train_bt = bt_model.zero_shot(train_ecfp, train_emb)

    model = XGBClassifier(n_estimators=1000, n_jobs=-1, random_state=42)
    model.fit(train_bt, train_y)

    return model,


def main():
    dt_str = "14062024_0910"
    bt_model = BarlowTwins()
    bt_model.load_model(f"./stash/{dt_str}")

    train_ecfp, train_emb, train_labels = load_dataset()
    print("Data loaded")
    model, train_bt, train_y = train((train_ecfp, train_emb, train_labels), bt_model)
    print("Model trained")

    model.save_model(f"./{dt_str}_barlowdti_xxl_model.json")
    # joblib.dump(model, f"./{dt_str}_barlowdti_xxl_model.pkl")
    print("Model saved")


if __name__ == "__main__":
    main()
