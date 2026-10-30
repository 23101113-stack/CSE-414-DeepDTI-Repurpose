import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from lifelines.utils import concordance_index

# Search path-e project root directory jog kora
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dataset import get_dataloaders
from src.models import DeepDTI

def generate_evaluation_plots():
    os.makedirs("results", exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Weights file khuje pawa (best_davis_model.pt ba best_model.pt)
    weight_path = os.path.join("weights", "best_davis_model.pt")
    if not os.path.exists(weight_path):
        weight_path = os.path.join("weights", "best_model.pt")

    if not os.path.exists(weight_path):
        print(f"Error: Weights file khuje pawa jayni: {weight_path}")
        return

    print(f"Loading weights from: {weight_path}")
    model = DeepDTI().to(device)
    model.load_state_dict(torch.load(weight_path, weights_only=True))
    model.eval()

    # 2. DAVIS dataset load kora
    data_path = os.path.join("data", "raw", "davis_all.csv")
    if not os.path.exists(data_path):
        data_path = "davis_all.csv"

    if not os.path.exists(data_path):
        print(f"Error: DAVIS dataset file khuje pawa jayni: {data_path}")
        return

    print("Loading test dataset...")
    _, _, test_loader = get_dataloaders(csv_path=data_path, batch_size=64)

    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            target_seq = batch.target_seq.squeeze(1).to(device)
            preds = model(batch.x, batch.edge_index, batch.batch, target_seq)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(batch.y.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    ci = concordance_index(all_targets, all_preds)
    r_val, _ = pearsonr(all_targets, all_preds)
    mse = np.mean((all_targets - all_preds) ** 2)

    # 3. Regression Plot toiri kora
    plt.figure(figsize=(7, 6))
    sns.regplot(
        x=all_targets, 
        y=all_preds, 
        scatter_kws={'alpha': 0.35, 'color': '#1f77b4'}, 
        line_kws={'color': '#d62728', 'linewidth': 2}
    )
    
    plt.title("DeepDTI: Ground Truth vs. Predicted Binding Affinity ($pK_d$)", fontsize=12, pad=12)
    plt.xlabel("Actual $pK_d$ Affinity", fontsize=11)
    plt.ylabel("Predicted $pK_d$ Affinity", fontsize=11)
    
    metrics_text = f"MSE: {mse:.4f}\nCI: {ci:.4f}\nPearson R: {r_val:.4f}"
    plt.gca().text(
        0.05, 0.92, metrics_text, 
        transform=plt.gca().transAxes,
        fontsize=10, 
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    )
    
    plt.tight_layout()
    output_path = os.path.join("results", "davis_test_evaluation.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Evaluation plot toiri hoye save hoyeche: {output_path}")

if __name__ == "__main__":
    generate_evaluation_plots()