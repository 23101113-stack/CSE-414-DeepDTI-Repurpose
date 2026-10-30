import os
import torch
import pandas as pd
from torch_geometric.data import Data, Batch
from src.models import DeepDTI
from src.dataset import smiles_to_graph, seq_to_tokens

# Example Pathogen Target Sequence: Dengue Virus NS5 Polymerase domain (fragment)
DENGUE_NS5_SEQ = (
    "GGNEGVVDKLFRKVKVRGEAGKFLDLSLPTAAGEDSEVVLPVSRVLPGEVFLVRGKEVFE"
    "QKKKVRGEAGKFLDLSLPTAAGEDSEVVLPVSRVLPGEVFLVRGKEVFEQKLKKVRGEA"
    "GTLGEVFEQKVRGEAGKFLDLSLPTAAGEDSEVVLPVSRVLPGEVFLVRGKEVFEQKLK"
)

# Sample FDA-Approved Molecules (Representative Library)
SAMPLE_DRUGS = [
    {"name": "Sofosbuvir", "smiles": "CC(C)OC(=O)C(C)NP(=O)(OCC1C(C(C(O1)N2C=CC(=O)NC2=O)(C)F)O)OC3=CC=CC=C3"},
    {"name": "Remdesivir", "smiles": "CCC(CC)COC(=O)C(C)NP(=O)(OCC1C(C(C(O1)(C#N)C2=CC=C3N2N=CN=C3N)O)O)OC4=CC=CC=C4"},
    {"name": "Ribavirin", "smiles": "C1C(C(C(O1)N2C=NC(=N2)C(=O)N)O)O"},
    {"name": "Favipiravir", "smiles": "C1=C(N=C(C(=O)N1)C(=O)N)F"},
    {"name": "Doxycycline", "smiles": "CC1C2CC3C(C(=O)C(=C(C3(C)O)O)C(=O)N)C(=O)C2=C(C4=C1C=CC=C4O)O"},
    {"name": "Chloroquine", "smiles": "CCN(CC)CCCC(C)NC1=C2C=CC(=CC2=NC=C1)Cl"},
    {"name": "Ivermectin", "smiles": "CCC(C)C1CC(CC2CC(CC=C(C(C(C=CC=C3C(O2)C(=O)OC4C3CC(C(=C4)C)O)C)O)C)O1)OC5CC(C(C(O5)C)O)OC6CC(C(C(O6)C)O)OC"}
]

def run_virtual_screening():
    os.makedirs("results", exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = DeepDTI().to(device)
    model.load_state_dict(torch.load("weights/best_model.pt", weights_only=True))
    model.eval()

    target_tensor = seq_to_tokens(DENGUE_NS5_SEQ, max_len=1000).unsqueeze(0).to(device)

    results = []
    print("Screening candidate small molecules against Dengue NS5 Target...")

    with torch.no_grad():
        for drug in SAMPLE_DRUGS:
            graph_data = smiles_to_graph(drug["smiles"])
            if graph_data[0] is None:
                continue

            x, edge_index = graph_data
            data = Data(x=x.to(device), edge_index=edge_index.to(device))
            batch = Batch.from_data_list([data])

            # Predict continuous pKd affinity
            pred_pkd = model(batch.x, batch.edge_index, batch.batch, target_tensor)
            score = pred_pkd.item()

            results.append({
                "Drug_Name": drug["name"],
                "SMILES": drug["smiles"],
                "Predicted_pKd": round(score, 4)
            })

    # Sort descending by predicted affinity
    df_results = pd.DataFrame(results).sort_values(by="Predicted_pKd", ascending=False).reset_index(drop=True)
    
    output_file = "results/repurposing_screening_results.csv"
    df_results.to_csv(output_file, index=False)
    
    print("\n--- Top Repurposed Lead Candidates ---")
    print(df_results[["Drug_Name", "Predicted_pKd"]])
    print(f"\nSaved screening ranks to: {output_file}")

if __name__ == "__main__":
    run_virtual_screening()