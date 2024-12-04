import torch.nn as nn
from model.CustomTransformerEncoder import CustomTransformerEncoder


# Hybrid Model with R(2+1)D + Swin Transformer
class HybridModel(nn.Module):
    def __init__(self, num_classes=10, embed_dim=512, num_heads=8, hidden_dim=1024, num_layers=6):
        super(HybridModel, self).__init__()

        # 3D CNN
        self.cnn3d = nn.Sequential(
            nn.Conv3d(in_channels=13, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=1, padding=1),
            nn.Conv3d(64, 128, (3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2), stride=1, padding=1),
            nn.Conv3d(128, 256, (3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 16, 16))
        )

        self.flatten_dim = 256 * 16 * 16
        self.fc_cnn3d = nn.Linear(self.flatten_dim, embed_dim)

        # Custom Encoder transformer
        self.transformer_encoder = CustomTransformerEncoder(embed_dim=embed_dim, num_heads=num_heads,
                                                            hidden_dim=hidden_dim, num_layers=num_layers)

        # Output layer
        self.fc_out = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # print("Input shape: ", x.shape)
        x = x.unsqueeze(2)

        x = self.cnn3d(x)
        x = x.view(x.size(0), -1)

        x = self.fc_cnn3d(x).unsqueeze(1)
        # print("After 3D CNN shape: ", x.shape)

        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        # print("After TransformerEncoder shape: ", x.shape)

        output = self.fc_out(x)
        return output
