import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from typing import *
import numpy as np
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


def try_or_none(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except:
        return None


def get_smiles(mols: List[rdkit.Chem.rdchem.Mol]) -> List[str]:
    """
    Gets list of smiles from list of rdkit molecules
    """
    return [Chem.MolToSmiles(x) for x in mols]


def get_mols(smiles: List[str]) -> List[rdkit.Chem.rdchem.Mol]:
    """
    Gets list of rdkit molecules from list of smiles
    """
    return [Chem.MolFromSmiles(x) for x in smiles]


def get_fp(
    mols: List[rdkit.Chem.rdchem.Mol],
    radius: int = 2,
    nBits: int = 1024,
    useFeatures: bool = False,
):
    """
    Computes ECFP/FCFP from list of RDKIT mols
    """

    output = np.empty(len(mols), dtype=object)

    for i, mol in enumerate(mols):
        output[i] = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=radius,
            nBits=nBits,
            useFeatures=useFeatures,
        )

    return output


def store_fp(fps: List, nBits: int = 1024):
    """
    Stores list of RDKIT sparse vectors in numpy array using C data structures
    """

    array = np.empty((len(fps), nBits), dtype=np.float32)
    for i in range(len(array)):
        DataStructs.ConvertToNumpyArray(fps[i], array[i])

    return array
