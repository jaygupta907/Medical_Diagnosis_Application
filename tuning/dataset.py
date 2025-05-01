import io
import sqlite3
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os

class ImageDatabaseDataset(Dataset):
    """
    PyTorch Dataset for loading images and corrected labels directly from an SQLite database.
    Only samples with a non-NULL corrected_label are loaded.
    """
    def __init__(self, db_path, table_name, transform=None):
        """
        Initialize the dataset by connecting to the database and loading the data.

        Args:
            db_path (Path or str): Path to the SQLite database file.
            table_name (str): Name of the table containing image and label data.
            transform (callable, optional): Transformations to apply to images.
        """
        self.db_path = db_path
        self.table_name = table_name
        self.conn = sqlite3.connect(str(self.db_path))
        self.cursor = self.conn.cursor()
        self.transform = transform

        # Fetch only entries where corrected_label is NOT NULL
        query = f"""
            SELECT image_data, predicted_label, corrected_label 
            FROM {self.table_name}
            WHERE corrected_label IS NOT NULL
        """
        self.cursor.execute(query)
        self.data = self.cursor.fetchall()

        if len(self.data) == 0:
            raise ValueError(f"No corrected labels found in table {self.table_name}")

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve a single sample (image and label) by index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (transformed image tensor, integer label)
        """
        img_bytes, predicted_label, corrected_label = self.data[idx]

        # Load image from bytes
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Always use corrected_label (since we filtered)
        label = corrected_label

        # Map label string to int
        label_map = {"COVID19": 0, "NORMAL": 1, "PNEUMONIA": 2}
        label = label_map.get(label, -1)  # Default to -1 if label is unknown

        return image, label

    def __del__(self):
        """
        Ensure the database connection is closed when the dataset object is deleted.
        """
        try:
            self.conn.close()
        except AttributeError:
            pass

class ImageDatabaseDataLoader:
    """
    Wrapper class to create a PyTorch DataLoader for the ImageDatabaseDataset.
    """
    def __init__(self, batch_size=32, shuffle=True, transform=None, table_name='prediction', db_path=None):
        """
        Initialize the DataLoader with the specified settings.

        Args:
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the dataset.
            transform (callable, optional): Transformations to apply to images.
            table_name (str): Name of the table containing image and label data.
            db_path (Path or str, optional): Path to the database file. Defaults to ./uploads/predictions.db.
        """
        self.db_path = db_path

        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database not found at {self.db_path}")

        self.table_name = table_name

        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        self.transform = transform

        self.dataset = ImageDatabaseDataset(
            db_path=self.db_path,
            table_name=self.table_name,
            transform=self.transform
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )

    def get_dataloader(self):
        """
        Return the created PyTorch DataLoader.

        Returns:
            DataLoader: DataLoader for iterating over the dataset.
        """
        return self.dataloader
