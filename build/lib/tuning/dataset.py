import io
import sqlite3
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageDatabaseDataset(Dataset):
    def __init__(self, db_path, table_name, transform=None):
        self.db_path = db_path
        self.table_name = table_name
        self.conn = sqlite3.connect(str(self.db_path))
        self.cursor = self.conn.cursor()
        self.transform = transform

        # Fetch all data
        query = f"SELECT image_data, predicted_label, corrected_label FROM {self.table_name}"
        self.cursor.execute(query)
        self.data = self.cursor.fetchall()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_bytes, predicted_label, corrected_label = self.data[idx]

        # Load image from bytes
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Choose corrected_label if available, otherwise predicted_label
        label = corrected_label if corrected_label is not None else predicted_label

        # You can choose to map string labels to integers for classification
        # If label is a string and you need integer labels for training, you can map it like this:
        label_map = {"COVID19": 0, "NORMAL": 1, "PNEUMONIA": 2}  # Example label mapping
        label = label_map.get(label, -1)  # Default to -1 if label is not in the map

        return image, label

    def __del__(self):
        self.conn.close()

class ImageDatabaseDataLoader:
    def __init__(self, batch_size=32, shuffle=True, transform=None, table_name='prediction'):  
        # Locate the database
        self.db_path = Path(__file__).resolve().parent.parent / "uploads" / "predictions.db"
        self.table_name = table_name

        # Define default transforms if none provided
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        self.transform = transform

        # Create dataset and dataloader
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
        return self.dataloader

# Example usage
if __name__ == "__main__":
    loader = ImageDatabaseDataLoader(batch_size=32, table_name='prediction')
    dataloader = loader.get_dataloader()

    for images, labels in dataloader:
        print(images.shape)  # Print the image shape
        print(labels)        # Print the labels (which are now integers or strings)
        break
