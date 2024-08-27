import numpy as np
import pandas as pd
import pickle
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


SMILES_COL = "Ligand"
PROTEIN_COL = "Protein"
LABEL_COL = "classification_label"


def get_ecfps(df, smiles_col=SMILES_COL):
    """Generate ECFPs from SMILES in the given DataFrame."""
    mols = [Chem.MolFromSmiles(smi) for smi in df[smiles_col]]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in mols]
    return np.array(fps)


def get_emb(df, protein_col=PROTEIN_COL, model="prost_t5"):
    """Generate embeddings from Proteins in the given DataFrame."""
    emb = encode_sequences(df[protein_col].tolist(), model)
    return np.array([np.array(x) for x in emb])


def load_dataset(dataset_name):
    """Load train, test, and validation sets for a given dataset."""
    base_path = f"../dataset/{dataset_name}/"
    train = pd.read_csv(base_path + "train.csv").dropna(subset=[SMILES_COL, PROTEIN_COL, LABEL_COL])
    test = pd.read_csv(base_path + "test.csv").dropna(subset=[SMILES_COL, PROTEIN_COL, LABEL_COL])
    valid = pd.read_csv(base_path + "valid.csv").dropna(subset=[SMILES_COL, PROTEIN_COL, LABEL_COL])
    return train, test, valid


def load_and_precompute_dataset_features(dataset_names):
    dataset_features = {}
    for dataset_name in dataset_names:
        print(f"Precomputing features for {dataset_name}")
        train, test, valid = load_dataset(dataset_name)

        train_ecfp, train_emb = get_ecfps(train), get_emb(train)
        valid_ecfp, valid_emb = get_ecfps(valid), get_emb(valid)
        test_ecfp, test_emb = get_ecfps(test), get_emb(test)

        train_labels = train[LABEL_COL].values
        valid_labels = valid[LABEL_COL].values
        test_labels = test[LABEL_COL].values

        dataset_features[dataset_name] = {
            "train": {"ecfp": train_ecfp, "emb": train_emb, "labels": train_labels},
            "valid": {"ecfp": valid_ecfp, "emb": valid_emb, "labels": valid_labels},
            "test": {"ecfp": test_ecfp, "emb": test_emb, "labels": test_labels}
        }
    return dataset_features


def combine_features(ecfp, emb):
    """Combine ECFP and embedding features."""
    return np.concatenate([ecfp, emb], axis=1)


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


def test_xgb_model(train_features, train_labels, test_features, test_labels, seed, name, params=None):
    if params is None:
        params = {}
    model = XGBClassifier(random_state=seed, n_jobs=64, **params)
    model.fit(train_features, train_labels)
    if seed == 0:
        name = name.replace("/", "_")
        model.save_model(f"xgb_models/xgb_model_{name}_{seed}.json")
    preds = model.predict_proba(test_features)[:, 1]
    roc = roc_auc_score(test_labels, preds)
    pr = average_precision_score(test_labels, preds)
    return {"ROC": roc, "PR": pr}


def opt_xgb_model(train_features, train_labels, valid_features, valid_labels, seed, name):
    model = XGBClsOptuna(seed=seed)
    model.optimize(train_features, train_labels, valid_features, valid_labels, name=name)
    # model.best_params["n_estimators"] = model.max_n_estimators
    return model.best_params


def benchmark(precomputed_features, name, bt_model, seeds: list = None):
    """Benchmark performance across models for a given dataset, across different seeds."""
    train_ecfp = precomputed_features["train"]["ecfp"]
    valid_ecfp = precomputed_features["valid"]["ecfp"]
    test_ecfp = precomputed_features["test"]["ecfp"]

    train_emb = precomputed_features["train"]["emb"]
    valid_emb = precomputed_features["valid"]["emb"]
    test_emb = precomputed_features["test"]["emb"]

    train_bt = bt_model.zero_shot(train_ecfp, train_emb)
    valid_bt = bt_model.zero_shot(valid_ecfp, valid_emb)
    test_bt = bt_model.zero_shot(test_ecfp, test_emb)

    y_train = precomputed_features["train"]["labels"]
    y_val = precomputed_features["valid"]["labels"]
    y_test = precomputed_features["test"]["labels"]

    train_combined = combine_features(train_ecfp, train_emb)
    valid_combined = combine_features(valid_ecfp, valid_emb)
    test_combined = combine_features(test_ecfp, test_emb)

    print(f"Optimizing XGB models for {name} combined.")
    xgb_best_params = opt_xgb_model(
        train_combined,
        y_train,
        valid_combined,
        y_val,
        seeds[0],
        name + "_combined"
    )

    print(f"Optimizing XGB models for {name} BT.")
    bt_best_params = opt_xgb_model(
        train_bt,
        y_train,
        valid_bt,
        y_val,
        seeds[0],
        name + "_bt"
    )

    results = {}
    for seed in seeds:
        seed_results = {}

        seed_results["XGB"] = test_xgb_model(
            train_combined,
            y_train,
            test_combined,
            y_test,
            seed,
            name + "_combined"
        )

        seed_results["XGB_Opt"] = test_xgb_model(
            train_combined,
            y_train,
            test_combined,
            y_test,
            seed,
            name + "_combined_optimized",
            xgb_best_params
        )

        seed_results["BT_XGB"] = test_xgb_model(
            train_bt,
            y_train,
            test_bt,
            y_test,
            seed,
            name + "_bt"
        )

        seed_results["BT_XGB_Opt"] = test_xgb_model(
            train_bt,
            y_train,
            test_bt,
            y_test,
            seed,
            name + "_bt_optimized",
            bt_best_params
        )

        results[seed] = seed_results

    best_params = {
        "XGB": xgb_best_params,
        "BT_XGB": bt_best_params
    }

    return results, best_params


def main():
    datasets = [
        "nature_mach_intel/BindingDB/cluster",
        "nature_mach_intel/BindingDB/protein",
        "nature_mach_intel/BindingDB/random",
        "nature_mach_intel/BindingDB/scaffold",

        "nature_mach_intel/BioSNAP/cluster",
        "nature_mach_intel/BioSNAP/protein",
        "nature_mach_intel/BioSNAP/random",
        "nature_mach_intel/BioSNAP/scaffold",

        "nature_mach_intel/Human/protein",
        "nature_mach_intel/Human/random",
        "nature_mach_intel/Human/scaffold",
    ]

    # Precompute features
    RECOMPUTE = True
    AA_EMB = "_prost"

    AA_EMB += "_nature_mi"

    if f"precomputed_data{AA_EMB}.pkl" in os.listdir(".") and not RECOMPUTE:
        precomputed_data = pickle.load(open(f"precomputed_data{AA_EMB}.pkl", "rb"))
        precomputed_data = {k: v for k, v in precomputed_data.items() if k in datasets}
        print("Features loaded.")
    else:
        precomputed_data = load_and_precompute_dataset_features(datasets)
        print("Features precomputed.")
        # Save features
        pickle.dump(precomputed_data, open(f"precomputed_data{AA_EMB}.pkl", "wb"))
        print("Features saved.")

    for dt_str in ["14062024_0910", "02082024_0755"]:
        bt_model = BarlowTwins()
        bt_model.load_model(f"./stash/{dt_str}")

        all_results = {}
        all_params = {}
        seeds = list(range(5))
        # seeds = [0]
        for dataset_name, features in tqdm(precomputed_data.items(), desc="Benchmarking Datasets"):
            name = dataset_name + "_" + dt_str
            all_results[dataset_name], all_params[dataset_name] = benchmark(features, name, bt_model, seeds)
        
        # Flatten the results for easier CSV storage
        flat_results = []
        for dataset, seed_results in all_results.items():
            for seed, model_results in seed_results.items():
                for model_name, metrics in model_results.items():
                    flat_results.append({
                        "Dataset": dataset.replace("/", "_"),
                        "Seed": seed,
                        "Model": model_name,
                        "ROC": metrics["ROC"],
                        "PR": metrics["PR"]
                    })

        pd.DataFrame(flat_results).to_csv(f"results/all_results_detailed_{dt_str + '_' + AA_EMB}.csv")
        pd.DataFrame(all_params).to_csv(f"results/all_params_{dt_str + '_' + AA_EMB}.csv")


if __name__ == "__main__":
    main()
