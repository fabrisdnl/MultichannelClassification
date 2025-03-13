import torch.nn as nn
from model.AdaptedTransformerEncoder import AdaptedTransformerEncoder


# Hybrid Model con fix alla CNN 3D
class AdaptedHybridModel(nn.Module):
    def __init__(self, num_channels=13, num_classes=10, embed_dim=512, num_heads=8, hidden_dim=1024, num_layers=6):
        super(AdaptedHybridModel, self).__init__()

        # CNN 3D (Fixata per evitare depth=0)
        self.cnn3d = nn.Sequential(
            nn.Conv3d(in_channels=num_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64, eps=1e-3),
            nn.Tanh(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),  # Mantiene depth invariata

            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128, eps=1e-3),
            nn.Tanh(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),  # Evita depth=0

            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256, eps=1e-3),
            nn.Tanh(),
            nn.AdaptiveAvgPool3d((1, 8, 8))
        )

        # Dimensionalità dopo la CNN
        self.flatten_dim = 256 * 8 * 8
        self.fc_cnn3d = nn.Linear(self.flatten_dim, embed_dim)

        # Custom Transformer Encoder con Positional Encoding
        self.transformer_encoder = AdaptedTransformerEncoder(embed_dim=embed_dim, num_heads=num_heads,
                                                            hidden_dim=hidden_dim, num_layers=num_layers)

        self.input_norm = nn.LayerNorm(embed_dim, eps=1e-3)

        # Output layer
        self.fc_out = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = x.unsqueeze(2)  # Espansione per compatibilità con CNN 3D
        x = self.cnn3d(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_cnn3d(x).unsqueeze(1)
        x = self.input_norm(x)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)  # Rimuove dimensione extra
        return self.fc_out(x)
