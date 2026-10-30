import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

# 1. Drug GNN Branch (Chemical Graph Representation)
class DrugGNN(nn.Module):
    def __init__(self, in_channels=5, hidden_dim=128, out_dim=128):
        super(DrugGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch):
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = self.relu(self.conv3(x, edge_index))
        # Global graph pooling -> shape: (batch_size, out_dim)
        h_d = global_mean_pool(x, batch)
        return h_d

# 2. Target Protein Sequence Branch (Multi-Scale CNN + BiLSTM)
class TargetBranch(nn.Module):
    def __init__(self, vocab_size=25, embed_dim=128, num_filters=64, lstm_hidden=64, out_dim=128):
        super(TargetBranch, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Multi-scale 1D CNN (kernel sizes 4, 8, 12)
        self.conv4 = nn.Conv1d(embed_dim, num_filters, kernel_size=4, padding='same')
        self.conv8 = nn.Conv1d(embed_dim, num_filters, kernel_size=8, padding='same')
        self.conv12 = nn.Conv1d(embed_dim, num_filters, kernel_size=12, padding='same')
        
        cnn_out_dim = num_filters * 3
        # Bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(lstm_hidden * 2, out_dim)
        self.relu = nn.ReLU()

    def forward(self, target_seq):
        # target_seq shape: (batch_size, seq_len)
        x = self.embedding(target_seq)               # (batch_size, seq_len, embed_dim)
        x = x.permute(0, 2, 1)                      # (batch_size, embed_dim, seq_len)
        
        c4 = self.relu(self.conv4(x))
        c8 = self.relu(self.conv8(x))
        c12 = self.relu(self.conv12(x))
        
        cnn_feat = torch.cat([c4, c8, c12], dim=1)  # (batch_size, num_filters*3, seq_len)
        cnn_feat = cnn_feat.permute(0, 2, 1)        # (batch_size, seq_len, num_filters*3)
        
        lstm_out, _ = self.bilstm(cnn_feat)         # (batch_size, seq_len, lstm_hidden*2)
        h_t, _ = torch.max(lstm_out, dim=1)         # Max-pooling over sequence
        h_t = self.relu(self.fc(h_t))               # (batch_size, out_dim)
        return h_t

# 3. Cross-Attention Fusion Layer
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=128):
        super(CrossAttentionFusion, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h_d, h_t):
        # Reshape to sequence length of 1: (batch_size, 1, dim)
        q = h_d.unsqueeze(1)
        k = v = h_t.unsqueeze(1)
        attn_out, _ = self.multihead_attn(q, k, v)
        fused = self.norm(q + attn_out).squeeze(1)   # (batch_size, dim)
        return fused

# 4. Full DeepDTI-Repurpose Framework
class DeepDTI(nn.Module):
    def __init__(self, drug_dim=128, target_dim=128, fused_dim=128):
        super(DeepDTI, self).__init__()
        self.drug_encoder = DrugGNN(in_channels=5, hidden_dim=128, out_dim=drug_dim)
        self.target_encoder = TargetBranch(vocab_size=25, embed_dim=128, out_dim=target_dim)
        self.fusion = CrossAttentionFusion(dim=fused_dim)
        
        # Dense MLP Head
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(64, 1)  # Predicted continuous pKd
        )

    def forward(self, x, edge_index, batch, target_seq):
        h_d = self.drug_encoder(x, edge_index, batch)
        h_t = self.target_encoder(target_seq)
        fused = self.fusion(h_d, h_t)
        out = self.mlp(fused)
        return out.squeeze(-1)

if __name__ == "__main__":
    # Test model forward pass
    model = DeepDTI()
    print("Model initialized successfully!")
    print(model)