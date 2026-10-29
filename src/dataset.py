import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from rdkit import Chem
from sklearn.model_selection import train_test_split

# 1. Standard 20 Amino Acid mapping dictionary
AMINO_ACID_MAP = {
    'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
    'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
    'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
    'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20
}

def seq_to_tokens(sequence, max_len=1000):
    """Protein sequence ke integer token vector e convert kore."""
    tokens = [AMINO_ACID_MAP.get(char, 0) for char in sequence[:max_len]]
    if len(tokens) < max_len:
        tokens = tokens + [0] * (max_len - len(tokens))  # Padding
    return torch.tensor(tokens, dtype=torch.long)

def smiles_to_graph(smiles):
    """SMILES theke Molecular Graph (Node & Edge features) toiri kore."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node Features: Atom atomic number, aromaticity, degree
    node_features = []
    for atom in mol.GetAtoms():
        feat = [
            atom.GetAtomicNum(),
            int(atom.GetIsAromatic()),
            atom.GetTotalDegree(),
            atom.GetFormalCharge(),
            int(atom.GetHybridization())
        ]
        node_features.append(feat)
    x = torch.tensor(node_features, dtype=torch.float)

    # Edge Features & Connectivity
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])  # Undirected graph

    if len(edge_indices) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    return x, edge_index

class DTIDataset(Dataset):
    def __init__(self, df, max_seq_len=1000):
        self.df = df.reset_index(drop=True)
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles = row['compound_iso_smiles']
        target_seq = row['target_sequence']
        affinity = float(row['affinity'])

        # Molecular Graph
        x, edge_index = smiles_to_graph(smiles)

        # Protein Sequence
        seq_tensor = seq_to_tokens(target_seq, max_len=self.max_seq_len)

        # PyG Data Object
        data = Data(
            x=x,
            edge_index=edge_index,
            target_seq=seq_tensor.unsqueeze(0),
            y=torch.tensor([affinity], dtype=torch.float)
        )
        return data

def get_dataloaders(csv_path="data\raw\davis_all.csv", batch_size=64):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['compound_iso_smiles', 'target_sequence', 'affinity'])

    # 80% Train, 10% Validation, 10% Test split
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    train_dataset = DTIDataset(train_df)
    val_dataset = DTIDataset(val_df)
    test_dataset = DTIDataset(test_df)

    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders("davis_all.csv")
    print(f"Dataset Loaded Successfully! Train batches: {len(train_loader)}")