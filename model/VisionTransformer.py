import torch
import torch.nn as nn


class VisionTransformer(nn.Module):
    def __init__(self, img_size=64, patch_size=8, num_classes=10, dim=256, depth=6, heads=8, mlp_dim=512, channels=13):
        super(VisionTransformer, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = channels * patch_size * patch_size
        self.dim = dim

        self.patch_embeddings = nn.Linear(self.patch_dim, dim)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_patches, -1)
        x = self.patch_embeddings(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.position_embeddings

        x = self.transformer_encoder(x)

        cls_token_final = x[:, 0]
        output = self.fc(cls_token_final)
        return output