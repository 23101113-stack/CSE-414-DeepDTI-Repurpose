import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from lifelines.utils import concordance_index
from scipy.stats import pearsonr

# Import dataset loader and model
from dataset import get_dataloaders
from models import DeepDTI

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        batch = batch.to(device)
        target_seq = batch.target_seq.squeeze(1).to(device)
        y = batch.y.to(device)

        optimizer.zero_grad()
        preds = model(batch.x, batch.edge_index, batch.batch, target_seq)
        loss = criterion(preds, y)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * len(y)

    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            target_seq = batch.target_seq.squeeze(1).to(device)
            y = batch.y.to(device)

            preds = model(batch.x, batch.edge_index, batch.batch, target_seq)
            loss = criterion(preds, y)

            total_loss += loss.item() * len(y)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(y.cpu().numpy().flatten())

    mse = total_loss / len(dataloader.dataset)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Concordance Index & Pearson Correlation
    ci = concordance_index(all_targets, all_preds)
    pearson_r, _ = pearsonr(all_targets, all_preds)

    return mse, ci, pearson_r

def main():
    # 1. Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Load Data (ensure davis_all.csv path is correct)
    data_path = os.path.join("data", "raw", "davis_all.csv")
    if not os.path.exists(data_path):
        data_path = "kiba_all.csv"  # Check root directory

    print("Loading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(csv_path=data_path, batch_size=64)

    # 3. Model, Loss, Optimizer, Scheduler
    model = DeepDTI().to(device)
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 30
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    os.makedirs("weights", exist_ok=True)
    best_val_ci = 0.0

    print("Starting training...")
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_ci, val_r = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"| Train MSE: {train_loss:.4f} "
              f"| Val MSE: {val_loss:.4f} "
              f"| Val CI: {val_ci:.4f} "
              f"| Val Pearson: {val_r:.4f}")

        # Save model checkpoint with best Concordance Index
        if val_ci > best_val_ci:
            best_val_ci = val_ci
            torch.save(model.state_dict(), "weights/best_model.pt")
            print(f"--> Saved best model checkpoint (Val CI: {val_ci:.4f})")

    # 4. Final Evaluation on Test Set
    print("\nRunning test set evaluation on best model...")
    model.load_state_dict(torch.load("weights/best_model.pt"))
    test_loss, test_ci, test_r = evaluate(model, test_loader, criterion, device)
    print("=" * 45)
    print(f"Final Test MSE       : {test_loss:.4f}")
    print(f"Final Test CI        : {test_ci:.4f}")
    print(f"Final Test Pearson R : {test_r:.4f}")
    print("=" * 45)

if __name__ == "__main__":
    main()