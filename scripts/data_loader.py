# scripts/data_loader.py

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset 
from stainlib.augmentation.stain_augment import RandomStainAugmentation
from stainlib.normalization.macenko import MacenkoNormalizer
import torch

def load_tile_dataset(data_dir, batch_size=32, shuffle=True, image_size=224, val_split=0.2):
    """Loads the tile dataset and splits into train and validation loaders with augmentation."""

    train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomAffine(degrees=30, shear=10, scale=(0.9, 1.1)),
        transforms.GaussianBlur(kernel_size=5),
        transforms.RandomErasing()
    ], p=0.7),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

    val_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
    #Load dataset without transforms
    base_dataset = datasets.ImageFolder(root=data_dir)

    # Split datasets into train/validation
    val_size = int(len(base_dataset) * val_split)
    train_size = len(base_dataset) - val_size

    train_indices, val_indices = torch.utils.data.random_split(
        range(len(base_dataset)),
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # --- Wrap subsets with their own transforms ---
    train_dataset = Subset(base_dataset, train_indices)
    val_dataset = Subset(base_dataset, val_indices)

    train_dataset.dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    val_dataset.dataset = datasets.ImageFolder(root=data_dir, transform=val_transform)

    # --- Create DataLoaders ---
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, base_dataset.class_to_idx

class StainNormalizedDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.normalizer = MacenkoNormalizer()
        
        # Pick a representative slide for target staining
        target_img = cv2.imread(image_paths[0])
        self.normalizer.fit(target_img)

    def __getitem__(self, index):
        img = cv2.imread(self.image_paths[index])
        img = self.normalizer.transform(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.image_paths)