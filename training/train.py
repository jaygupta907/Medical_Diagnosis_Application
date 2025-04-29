import torch
import torch.nn as nn
import torch.optim as optim
import logging
from model import resnet
from dataset import chestdataset
from tqdm import tqdm
import mlflow
from mlflow.models import infer_signature  
import argparse
import os

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Trainer class for handling training, validation, and testing of the model
class Trainer:
    """
    Trainer class for training, evaluating, and testing a model.

    Args:
        model (torch.nn.Module): Model to be trained.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        test_loader (DataLoader): DataLoader for test data.
        criterion (loss function): Loss function.
        optimizer (optimizer): Optimizer for model parameters.
        num_epochs (int): Number of epochs to train.
        device (str): Device to run the training on ("cuda" or "cpu").
        eval_frequency (int): Frequency (in epochs) to evaluate on validation set.

    Returns:
        None
    """
    def __init__(self, model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs=10, device="cuda", eval_frequency=2):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = device
        self.eval_frequency = eval_frequency
    
    # Method to train the model for the specified number of epochs
    def train(self):
        """
        Trains the model for the specified number of epochs.

        Args:
            None

        Returns:
            None
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

            if (epoch + 1) % self.eval_frequency == 0:
                logger.info("Evaluating on validation dataset")
                val_accuracy, val_loss = self.evaluate(self.val_loader)
                logger.info(f"Epoch [{epoch+1}/{self.num_epochs}] | Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}% | Val Accuracy: {val_accuracy:.2f}% | Val Loss: {val_loss:.4f}")
                mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
            else:
                logger.info(f"Epoch [{epoch+1}/{self.num_epochs}] | Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%")

    # Method to evaluate the model on a given dataloader (validation or test)
    def evaluate(self, loader):
        """
        Evaluates the model on the given dataloader.

        Args:
            loader (DataLoader): DataLoader for validation or test set.

        Returns:
            accuracy (float): Accuracy percentage.
            avg_loss (float): Average loss value.
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

    # Method to test the model performance on the test set
    def test(self):
        """
        Tests the model on the test set.

        Args:
            None

        Returns:
            test_accuracy (float): Accuracy on test dataset.
            test_loss (float): Loss on test dataset.
        """
        test_accuracy, test_loss = self.evaluate(self.test_loader)
        logger.info(f"Test Accuracy: {test_accuracy:.2f}%")
        return test_accuracy, test_loss

# Main function that sets up dataset, model, trainer, and MLflow logging
def main(args):
    """
    Main function to initialize datasets, model, trainer and start training.

    Args:
        args (Namespace): Command-line arguments containing training configurations.

    Returns:
        None
    """
    logger.info("Setting MLflow tracking URI and experiment.")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("Lung_Disease_Prediction")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    dataset = chestdataset(
        dataset_path="../datasets/chestxray/Data",
        batch_size=args.batch_size,
        apply_augmentation=True,
    )
    train_loader, val_loader, test_loader = dataset.get_dataloaders()

    model = resnet(in_planes=3, outputs=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        device=device,
        eval_frequency=args.eval_frequency
    )

    with mlflow.start_run(run_name=args.run_name) as run:

        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("lr", args.learning_rate)
        mlflow.log_param("epochs", args.num_epochs)

        logger.info("Training started...")
        trainer.train()

        trainer.test()

        model_path = "../model/trained_model.pt"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(trainer.model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")

        example_input, example_output = next(iter(train_loader))
        signature = infer_signature(example_input.cpu().numpy(), example_output.cpu().detach().numpy())
        mlflow.pytorch.log_model(model, artifact_path="model", signature=signature)

        run_id = run.info.run_id
        result = mlflow.register_model(
            model_uri=f"runs:/{run_id}/model",
            name="lung_disease_prediction_model"
        )

# Entry point: parse command-line arguments and start main()
if __name__ == "__main__":
    """
    Parses command-line arguments and starts the main training pipeline.

    Args:
        None (Arguments are parsed inside)

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description='mlflow logging for lung disease prediction')
    parser.add_argument('--run_name', type=str, default='run_1', help='Name of the run')
    parser.add_argument('--num_epochs', type=int, default=1, help='Epochs for model training')
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='Learning Rate for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--eval_frequency', type=int, default=1, help='Frequency for evaluation on validation dataset')
    args = parser.parse_args()

    main(args)
