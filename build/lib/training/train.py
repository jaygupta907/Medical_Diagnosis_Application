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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()



class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs=10, device="cuda",eval_frequency=2):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = device
        self.eval_frequency = eval_frequency
    
    def train(self):
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
                val_accuracy ,val_loss= self.evaluate(self.val_loader)
                logger.info(f"Epoch [{epoch+1}/{self.num_epochs}] | Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}% | Val Acccuracy: {val_accuracy:.2f}% | Val Loss: {val_loss:.4f}")
                mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
            else: 
                logger.info(f"Epoch [{epoch+1}/{self.num_epochs}] | Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%")

    
    def evaluate(self, loader):
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

    def test(self):
        test_accuracy ,test_loss= self.evaluate(self.test_loader)
        logger.info(f"Test Accuracy: {test_accuracy:.2f}%")
        return test_accuracy,test_loss



def main(args):
    logger.info("Setting MLflow tracking URI and experiment.")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("Lung_Disease_Prediction")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load dataset
    dataset = chestdataset(
        dataset_path="../datasets/chestxray/Data",
        batch_size=args.batch_size,
        apply_augmentation=True,
    )
    train_loader, val_loader, test_loader = dataset.get_dataloaders()

    # Initialize model
    model = resnet(in_planes=3,outputs=3)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Initialize trainer
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
        # Train the model
        logger.info("Training started...")
        trainer.train()

        # Test the model
        trainer.test()

        
        model_path = "../model/trained_model.pt"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(trainer.model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")

        example_input , example_output = next(iter(train_loader))
        signature = infer_signature(example_input.cpu().numpy(), example_output.cpu().detach().numpy())
        mlflow.pytorch.log_model(model, artifact_path="model", signature=signature)
        run_id = run.info.run_id
        result = mlflow.register_model(
            model_uri=f"runs:/{run_id}/model",
            name="lung_disease_prediction_model"
        )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='mlflow logging for lung disease prediction')
    parser.add_argument('--run_name', type=str, default='run_1', help='Name of the run')
    parser.add_argument('--num_epochs', type=int, default=1, help='Epochs for model training')
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='Learning Rate for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--eval_frequency', type=int, default=1, help='frequency for evaluation on validation dataset')
    args = parser.parse_args()
    main(args)
