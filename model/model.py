import sys
from typing import List
from tqdm import tqdm
import pandas as pd
import numpy as np
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import time
import requests
import joblib
# from bio_embeddings.embed import SeqVecEmbedder, ProtTransBertBFDEmbedder, ProtTransT5XLU50Embedder
from Bio import SeqIO
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import torch
from typing import *
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from xgboost import XGBClassifier, DMatrix
from lightgbm import LGBMClassifier

from barlow_twins import BarlowTwins

sys.path.append("../utils/")
from sequence import uniprot2sequence, encode_sequences


class DTIModel:
    def __init__(self, bt_model_path: str, gbm_model_path: str, encoder: str = "prost_t5", gbm_type: str = "xgbm"):
        self.bt_model = BarlowTwins()
        self.bt_model.load_model(bt_model_path)

        if gbm_type == "xgbm":
            self.gbm_type = "xgbm"
            self.gbm_model = XGBClassifier()
            self.gbm_model.load_model(gbm_model_path)
        elif gbm_type == "lgbm":
            self.gbm_type = "lgbm"
            self.gbm_model = joblib.load(gbm_model_path)
        else:
            raise ValueError(f"Invalid GBM type: {gbm_type}, oly 'xgbm' and 'lgbm' are supported")

        self.encoder = encoder

        self.smiles_cache = {}
        self.sequence_cache = {}

    def _encode_smiles(self, smiles: str, radius: int = 2, bits: int = 1024, features: bool = False):
        if smiles is None:
            return None
        # Check if the SMILES is already in the cache
        if smiles in self.smiles_cache:
            return self.smiles_cache[smiles]
        else:
            # Encode the SMILES and store it in the cache
            try:
                mol = Chem.MolFromSmiles(smiles)
                morgan = AllChem.GetMorganFingerprintAsBitVect(
                    mol,
                    radius=radius,
                    nBits=bits,
                    useFeatures=features,
                )
                morgan = np.array(morgan)
                self.smiles_cache[smiles] = morgan
                return morgan
            except Exception as e:
                print(f"Failed to encode SMILES: {smiles}")
                print(e)
                return None

    def _encode_smiles_mult(self, smiles: List[str], radius: int = 2, bits: int = 1024, features: bool = False):
        morgan = [self._encode_smiles(s, radius, bits, features) for s in smiles]
        return np.array(morgan)

    def _encode_sequence(self, sequence: str):
        # Clear torch cache
        torch.cuda.empty_cache()
        if sequence is None:
            return None
        # Check if the sequence is already in the cache
        if sequence in self.sequence_cache:
            return self.sequence_cache[sequence]
        else:
            # Encode the sequence and store it in the cache
            try:
                encoded_sequence = encode_sequences([sequence], encoder=self.encoder)
                self.sequence_cache[sequence] = encoded_sequence
                return encoded_sequence
            except Exception as e:
                print(f"Failed to encode sequence: {sequence}")
                print(e)
                return None

    def _encode_sequence_mult(self, sequences: List[str]):
        seq = [self._encode_sequence(sequence) for sequence in sequences]
        return np.array(seq)

    def __predict_pair(self, drug_emb: np.ndarray, target_emb: np.ndarray, pred_leaf: bool):
        if drug_emb.shape[0] < target_emb.shape[0]:
            drug_emb = np.tile(drug_emb, (len(target_emb), 1))
        elif len(drug_emb) > len(target_emb):
            target_emb = np.tile(target_emb, (len(drug_emb), 1))
        emb = self.bt_model.zero_shot(drug_emb, target_emb)

        if pred_leaf:
            if self.gbm_type == "xgbm":
                d_emb = DMatrix(emb)
                return self.gbm_model.get_booster().predict(d_emb, pred_leaf=True)
            else:
                return self.gbm_model.predict_proba(emb, pred_leaf=True)
        else:
            return self.gbm_model.predict_proba(emb)[:, 1]

    def predict(self, drug: List[str] or str, target: str, pred_leaf: bool = False):
        if isinstance(drug, str):
            drug_emb = self._encode_smiles(drug)
        else:
            drug_emb = self._encode_smiles_mult(drug)
        target_emb = self._encode_sequence(target)
        return self.__predict_pair(drug_emb, target_emb, pred_leaf)
    
    def get_leaf_weights(self):
        return self.gbm_model.get_booster().get_score(importance_type="weight")

    def _predict_fasta(self, drug: str, fasta_path: str):
        drug_emb = self._encode_smiles(drug)

        results = []
        # Extract targets from fasta
        for target in tqdm(SeqIO.parse(fasta_path, "fasta"), desc="Predicting targets"):
            target_emb = self._encode_sequence(str(target.seq))
            pred = self.__predict_pair(drug_emb, target_emb)
            results.append(
                {
                    "drug": drug,
                    "target": target.id,
                    "name": target.name,
                    "description": target.description,
                    "prediction": pred[0]
                }
            )
        return pd.DataFrame(results)

    def predict_fasta(self, drug: str, fasta_path: str, timeout_seconds: int = 120):
        def process_target(target, results):
            target_emb = self._encode_sequence(str(target.seq))
            pred = self.__predict_pair(drug_emb, target_emb)
            results.append({
                "drug": drug,
                "target": target.id,
                "name": target.name,
                "description": target.description,
                "prediction": pred[0]
            })

        drug_emb = self._encode_smiles(drug)
        results = []

        # First, count the total number of records for the progress bar
        total_records = sum(1 for _ in SeqIO.parse(fasta_path, "fasta"))

        # Extract targets from fasta with a properly initialized tqdm progress bar
        for target in tqdm(SeqIO.parse(fasta_path, "fasta"), total=total_records, desc="Predicting targets"):
            thread_results = []
            thread = threading.Thread(target=process_target, args=(target, thread_results))
            thread.start()
            thread.join(timeout_seconds)
            if thread.is_alive():
                print(f"Skipping target {target.id} due to timeout")
                continue
            results.extend(thread_results)

        return pd.DataFrame(results)

    def predict_uniprot(self, drug: List[str] or str, uniprot_id: str):
        return self.predict(drug, uniprot2sequence(uniprot_id))
