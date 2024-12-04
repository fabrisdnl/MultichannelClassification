import torch
import torch.nn as nn
from model.TransformerEncoderLayer import TransformerEncoderLayer


class LightweightViT(nn.Module):
    def __init__(self, img_size=64, patch_size=4, in_chans=13, num_classes=10, embed_dim=64, depth=6, num_heads=4, mlp_ratio=3.0, dropout=0.3):
        super(LightweightViT, self).__init__()

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.position_embeddings
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x[:, 0])
        x = self.fc(x)
        return x