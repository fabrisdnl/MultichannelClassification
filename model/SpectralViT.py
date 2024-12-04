import torch
import torch.nn as nn


class SpectralViT(nn.Module):
    def __init__(self, num_classes=10, emb_size=128, num_heads=4, depth=6):
        super(SpectralViT, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=1, out_channels=emb_size, kernel_size=(13, 3, 3), stride=(1, 2, 2))
        self.class_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.pos_embedding = nn.Parameter(torch.randn(1, (64 // 2) * (64 // 2) + 1, emb_size))
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=emb_size, nhead=num_heads)
            for _ in range(depth)
        ])
        self.spectral_attention = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size),
            nn.Sigmoid()
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, num_classes)
        )

    def forward(self, x):
        # Adding a channel dimension for the 3D convolution
        x = x.unsqueeze(1)
        # Applying the 3D convolution and removing the depth
        x = self.conv3d(x).squeeze(2)
        x = x.flatten(2).transpose(1, 2)
        b, n, _ = x.shape
        class_tokens = self.class_token.expand(b, -1, -1)
        x = torch.cat((class_tokens, x), dim=1)
        x += self.pos_embedding[:, :n+1]

        for layer in self.transformer_layers:
            x = layer(x)

        # Applying spectral attention
        x = self.spectral_attention(x[:, 0] * x[:, 0])
        return self.mlp_head(x)