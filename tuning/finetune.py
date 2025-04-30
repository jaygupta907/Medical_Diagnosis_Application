from tuning.dataset import ImageDatabaseDataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from training.model import resnet
from tqdm import tqdm
import mlflow
from mlflow.models import infer_signature  
import argparse
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class Trainer:
    """
    Trainer class to handle model training and evaluation.
    """
    def __init__(self, model, train_loader, criterion, optimizer, num_epochs=10, device="cuda"):
        """
        Initialize the Trainer.

        Args:
            model (torch.nn.Module): Model to train.
            train_loader (DataLoader): Training data loader.
            criterion (Loss): Loss function.
            optimizer (Optimizer): Optimizer for model training.
            num_epochs (int, optional): Number of training epochs. Defaults to 10.
            device (str, optional): Device to train on ('cuda' or 'cpu'). Defaults to 'cuda'.
        """
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = device
    
    def train(self):
        """
        Train the model and log metrics to MLflow.
        """
        self.model.to(self.device)
        
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss, correct, total = 0.0, 0, 0

            for images, labels in tqdm(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_loss = running_loss / total
            train_accuracy = 100 * correct / total

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)

            logger.info(f"Epoch [{epoch+1}/{self.num_epochs}] | Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%")
    
    def evaluate(self, loader):
        """
        Evaluate the model on a given dataset loader.

        Args:
            loader (DataLoader): DataLoader for evaluation data.

        Returns:
            tuple: (accuracy, average loss)
        """
        self.model.eval()
        correct, total = 0, 0
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(loader):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = running_loss / total
        accuracy = 100 * correct / total
        return accuracy, avg_loss

def main(args):
    """
    Main function to set up MLflow experiment, load dataset and model, train the model, and log results.

    Args:
        args (argparse.Namespace): Command line arguments for training configuration.
    """
    logger.info("Setting MLflow tracking URI and experiment.")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("Lung_Disease_Prediction")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load dataset
    dataset = ImageDatabaseDataLoader(batch_size=32, shuffle=True, transform=None, table_name='prediction',db_path="../uploads/predictions.db")
    train_loader = dataset.get_dataloader()

    # Initialize model
    model = resnet(in_planes=3, outputs=3)
    model_path = "../model/trained_model.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        model.load_state_dict(torch.load("../model/finetuned_model.pt", map_location=device))
    

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        device=device,
    )

    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("lr", args.learning_rate)
        mlflow.log_param("epochs", args.num_epochs)

        # Train the model
        logger.info("Training started...")
        trainer.train()

        # Save the fine-tuned model
        model_path = "../model/finetuned_model.pt"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(trainer.model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")

        # Log the model to MLflow with input-output signature
        example_input, example_output = next(iter(train_loader))
        signature = infer_signature(example_input.cpu().numpy(), example_output.cpu().detach().numpy())
        mlflow.pytorch.log_model(model, artifact_path="model", signature=signature)

        # Register the model in MLflow Model Registry
        run_id = run.info.run_id
        result = mlflow.register_model(
            model_uri=f"runs:/{run_id}/model",
            name="lung_disease_prediction_model_finetuned"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='mlflow logging for lung disease prediction')
    parser.add_argument('--run_name', type=str, default='tuning_run_1', help='Name of the run')
    parser.add_argument('--num_epochs', type=int, default=1, help='Epochs for model training')
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='Learning Rate for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    args = parser.parse_args()
    main(args)
