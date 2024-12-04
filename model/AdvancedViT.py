import torch
import torch.nn as nn


class AdvancedViT(nn.Module):
    def __init__(self, num_classes=10, dim=256, depth=12, heads=8, mlp_dim=512, dropout=0.3):
        super(AdvancedViT, self).__init__()
        self.embedding = nn.Linear(64 * 64 * 13, dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.classifier = nn.Sequential(
            nn.Linear(dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x