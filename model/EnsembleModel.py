import torch
import torch.nn as nn


class EnsembleModel(nn.Module):
    def __init__(self, cnn_model, vit_model, num_classes=10):
        super(EnsembleModel, self).__init__()
        self.cnn_model = cnn_model
        self.vit_model = vit_model
        self.fc = nn.Linear(2 * num_classes, num_classes)

    def forward(self, x):
        cnn_out = self.cnn_model(x)
        vit_out = self.vit_model(x)
        combined_out = torch.cat((cnn_out, vit_out), dim=1)
        return self.fc(combined_out)