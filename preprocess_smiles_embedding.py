import pandas as pd
import numpy as np
import torch as th
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

# Read SMILES
df = pd.read_csv("my_drug_with_smiles_Cdataset.csv")
df["smiles"] = df["smiles"].replace('nan', '').astype(str).fillna("")##
smiles_list = df["smiles"].tolist()##


def smiles_to_features(smiles):
    """Generate rich molecular features (without transformers)"""
    smiles = smiles.encode('ascii', errors='ignore').decode('ascii')##
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return np.zeros(2048 + 200, dtype=np.float32)##

    # 1. Morgan fingerprint (2048 dimensions)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    fp_arr = np.array(fp, dtype=np.float32)

    # 2. Molecular descriptors (200 dimensions)
    descriptors = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.NumAliphaticRings(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.NHOHCount(mol),
        Descriptors.NOCount(mol),
        Descriptors.NumHeteroatoms(mol),
        Descriptors.RingCount(mol),
        Descriptors.FractionCSP3(mol),
    ]

    # Pad to 200 dimensions
    while len(descriptors) < 200:
        descriptors.append(0.0)

    desc_arr = np.array(descriptors[:200], dtype=np.float32)
    return np.concatenate([fp_arr, desc_arr])


# Batch generation
print("Generating molecular features with RDKit...")
features = np.array([smiles_to_features(s) for s in smiles_list])
print(f"Features shape: {features.shape}")  # (num_drug, 2248)

# Save
smile_embeddings = th.FloatTensor(features)
th.save(smile_embeddings, "smile_embeddings.pt")
print(f"Saved to smile_embeddings.pt: {smile_embeddings.shape}")