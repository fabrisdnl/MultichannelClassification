from torch.utils.data import Dataset
from utils import utils


class EuroSATDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load the 13-band image
        img = utils.load_image_torch(img_path)

        # Permute the dimensions to [channels, height, width]
        img = img.permute(2, 0, 1)  # Change from [height, width, channels] to [channels, height, width]

        # Apply transformations
        if self.transform:
            img = self.transform(img)

        return img, label
