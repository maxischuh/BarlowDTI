import os
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch
from rdkit import Chem, DataStructs
import pandas as pd
import pickle as pkl
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys

sys.path.append("../utils/")
from parallel import *
from chem import *
from sequence import *


class Preprocessor:
    def __init__(
            self,
            path: str,
            radius: int = 2,
            n_bits: int = 1024,
            aa_embedding: str = "prottrans_t5_xl_u50",
            num_workers: int = 1,
    ):
        self.path = path
        self.radius = radius
        self.n_bits = n_bits
        self.aa_embedding = aa_embedding
        self.num_workers = num_workers

        self.data = None
        self.fp = None
        self.aa = None
        self.split = None
        self.label = None

        self.load_data()
        self.process_data()

    def load_data(self):
        if os.path.isfile(self.path):
            self.data = pd.read_csv(self.path, low_memory=False)
        else:
            raise ValueError("No data file found in the specified path")

    def process_data(self):
        if "smiles" not in self.data.columns:
            raise ValueError("No smiles column found in the data")
        if "sequence" not in self.data.columns:
            raise ValueError("No sequence column found in the data")

        smiles = self.data.smiles.tolist()
        seq = self.data.sequence.tolist()

        if "split" in self.data.columns:
            self.split = self.data.split.tolist()
        if "label" in self.data.columns:
            self.label = self.data.label.tolist()

        if self.num_workers > 1:
            mols = parallel(get_mols, self.num_workers, smiles)
            fps = parallel(get_fp, self.num_workers, mols, self.radius, self.n_bits)
        else:
            mols = get_mols(smiles)

            fps = get_fp(mols, self.radius, self.n_bits)

        self.fp = store_fp(fps, self.n_bits)
        self.aa = encode_sequences(seq, self.aa_embedding)

    def return_generator(
            self,
            device,
            batch_size: int = 512,
            include_negatives: bool = False,
            shuffle: bool = True,
            validation_split: float = None,
    ) -> (DataLoader, DataLoader):

        if self.split is None and self.label is None:
            print("No split or label columns found in the dataset")
            dataset = MolAADataset(device, self.fp, self.aa)
        elif self.split is not None:
            print("Splitting data into train and validation sets from the dataset without considering labels")
            train_fp, train_aa, val_fp, val_aa = [], [], [], []
            for i in range(len(self.fp)):
                if self.split[i] == "train":
                    train_fp.append(self.fp[i])
                    train_aa.append(self.aa[i])
                    
                elif self.split[i] == "val":
                    val_fp.append(self.fp[i])
                    val_aa.append(self.aa[i])
            
            train_dataset = MolAADataset(device, train_fp, train_aa)
            val_dataset = MolAADataset(device, val_fp, val_aa)

            print(f"Train: {len(train_fp)}, Validation: {len(val_fp)}")

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
            validation_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
            return train_loader, validation_loader
        
        else:
            print("Splitting data into train and validation sets from the dataset")
            train_fp, train_aa, val_fp, val_aa = [], [], [], []
            for i in range(len(self.fp)):
                if self.split[i] == "train":
                    if include_negatives and self.label[i] == 0:
                        train_fp.append(self.fp[i])
                        train_aa.append(self.aa[i] * -1)
                    elif self.label[i] == 1:
                        train_fp.append(self.fp[i])
                        train_aa.append(self.aa[i])
                elif self.split[i] == "val":
                    if include_negatives and self.label[i] == 0:
                        val_fp.append(self.fp[i])
                        val_aa.append(self.aa[i] * -1)
                    elif self.label[i] == 1:
                        val_fp.append(self.fp[i])
                        val_aa.append(self.aa[i])

            train_dataset = MolAADataset(device, train_fp, train_aa)
            val_dataset = MolAADataset(device, val_fp, val_aa)

            print(f"Train: {len(train_fp)}, Validation: {len(val_fp)}")

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
            validation_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
            return train_loader, validation_loader

        if validation_split is not None:
            print("Splitting data into train and validation by fractionation from the dataset")
            dataset_size = len(dataset)
            indices = list(range(dataset_size))
            split = int(np.floor(validation_split * dataset_size))
            if shuffle:
                np.random.shuffle(indices)
            train_indices, val_indices = indices[split:], indices[:split]

            train_sampler = SubsetRandomSampler(train_indices)
            valid_sampler = SubsetRandomSampler(val_indices)

            train_loader = DataLoader(
                dataset, batch_size=batch_size, sampler=train_sampler
            )
            validation_loader = DataLoader(
                dataset, batch_size=batch_size, sampler=valid_sampler
            )
            return train_loader, validation_loader

        else:
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
            return train_loader, None


class MolAADataset(Dataset):
    def __init__(self, device, mol, aa):
        self.mol = mol
        self.aa = aa
        self.device = device

    def __len__(self):
        """
        Method necessary for Pytorch training
        """
        return len(self.mol)

    def __getitem__(self, idx):
        """
        Method necessary for Pytorch training
        """
        mol_sample = torch.tensor(self.mol[idx], dtype=torch.float32)
        aa_sample = torch.tensor(self.aa[idx], dtype=torch.float32)

        mol_sample = mol_sample.to(self.device)
        aa_sample = aa_sample.to(self.device)

        return mol_sample, aa_sample
