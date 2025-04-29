import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class chestdataset:
    def __init__(self, dataset_path="datasets/chestxray/Data",
                 batch_size=32,
                 apply_augmentation=False):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.apply_augmentation = apply_augmentation

    def _get_transforms(self):
        # Define data transformations
        transform_list = [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
        
        if self.apply_augmentation:
            # Apply data augmentation techniques like rotation, translation, etc.
            augmentations = transforms.Compose([
                transforms.RandomRotation(30),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(translate=(0.1, 0.1),degrees=0),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            ])
            transform_list = [augmentations] + transform_list
        
        return transforms.Compose(transform_list)

    def get_dataloaders(self):
        transform = self._get_transforms()

        train_dir = os.path.join(self.dataset_path, "train")
        # Load full dataset
        full_train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)

        # Create sampler
        targets = [sample[1] for sample in full_train_dataset]
        class_counts = np.bincount(targets)
        class_weights = 1. / class_counts
        sample_weights = [class_weights[t] for t in targets]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(full_train_dataset, batch_size=self.batch_size, sampler=sampler)

        # Test dataset
        test_dir = os.path.join(self.dataset_path, "test")
        test_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader, test_loader
