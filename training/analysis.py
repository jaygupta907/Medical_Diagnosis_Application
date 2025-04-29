import os
import plotly.express as px
import logging
import argparse
from PIL import Image
import numpy as np
from sklearn.manifold import TSNE

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Argument parser for the script
parser = argparse.ArgumentParser(description='EDA Arguments')
parser.add_argument('--dataset_path', type=str, default='datasets/chestxray/Data')  # Dataset path
parser.add_argument('--analysis_path', type=str, default='analysis')  # Path to store analysis results
parser.add_argument('--image_size', type=int, default=128)  # Image size for processing
args = parser.parse_args()

# Create analysis directory if not exists
os.makedirs(args.analysis_path, exist_ok=True)

# Function to plot the image distribution in the dataset (train/test)
def plot_distribution():
    for split in ['train', 'test']:  # Loop over train and test splits
        split_path = os.path.join(args.dataset_path, split)  # Path for the split
        
        # Check if split directory exists
        if not os.path.isdir(split_path):
            logging.warning(f"{split_path} not found or is not a directory.")
            continue

        class_counts = {}  # Dictionary to store class-wise image counts
        for class_name in os.listdir(split_path):  # Iterate over class directories
            class_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_path):  # If class path is a directory
                # Count the number of image files in the class directory
                image_count = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                class_counts[class_name] = image_count  # Store class-wise image count

        class_names = list(class_counts.keys())  # List of class names
        image_counts = list(class_counts.values())  # List of image counts

        logging.info(f"{split.capitalize()} classes: {class_counts}")

        # Create a bar plot of image distribution per class
        fig = px.bar(
            x=class_names,
            y=image_counts,
            labels={'x': 'Class Name', 'y': 'Number of Images'},
            title=f'Image Distribution in {split.capitalize()} Set'
        )
        fig.update_layout(xaxis_tickangle=-45)

        # Save plot as PNG
        distribution_path = os.path.join(args.analysis_path,"distribution")
        os.makedirs(distribution_path,exist_ok=True)
        png_path = os.path.join(distribution_path, f"{split}_image_distribution.png")
        fig.write_image(png_path)
        logging.info(f"Saved plots for {split} at {png_path}")

    # Return paths for train/test image distribution plots
    return { "train": "/analysis/distribution/train_image_distribution.png", "test": "/analysis/distribution/test_image_distribution.png" }

# Function to perform t-SNE analysis on dataset (train/test)
def plot_tsne():
    for split in ['train', 'test']:  # Loop over train and test splits
        split_path = os.path.join(args.dataset_path, split)  # Path for the split
        if not os.path.isdir(split_path):  # Check if split directory exists
            logging.warning(f"{split_path} not found or is not a directory.")
            continue

        images = []  # List to store image data
        labels = []  # List to store labels
        class_image_counts = {}  # Dictionary to store count of images per class
        
        # Set max count of images per class for train and test splits
        if split == 'train':
            max_count = 400
        else:
            max_count = 100
        
        # Process each class directory
        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if not os.path.isdir(class_path):  # If not a directory, skip
                continue

            count = 0
            for img_file in os.listdir(class_path):
                if count >= max_count:  # Limit number of images processed per class
                    break

                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Process image files
                    img_path = os.path.join(class_path, img_file)
                    try:
                        # Open image, convert to grayscale, resize and flatten
                        img = Image.open(img_path).convert("L").resize((args.image_size, args.image_size))
                        img_array = np.asarray(img).flatten() / 255.0  # Normalize pixel values
                        images.append(img_array)
                        labels.append(class_name)
                        count += 1
                    except Exception as e:
                        logging.warning(f"Failed to process {img_path}: {e}")  # Handle exceptions

            class_image_counts[class_name] = count  # Store class-wise image count

        if not images:  # If no images found, skip t-SNE
            logging.warning(f"No images found in {split} for t-SNE.")
            continue

        logging.info(f"{split.capitalize()} samples used per class: {class_image_counts}")
        logging.info(f"Running t-SNE on {len(images)} total images from {split} set...")

        # Convert image data to NumPy array for t-SNE processing
        images_np = np.array(images)

        # Create and run t-SNE with specified parameters
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        tsne_result = tsne.fit_transform(images_np)

        # Create scatter plot of t-SNE results
        fig = px.scatter(
            x=tsne_result[:, 0],
            y=tsne_result[:, 1],
            color=labels,
            labels={"x": "t-SNE Dimension 1", "y": "t-SNE Dimension 2"},
            title=f"t-SNE Visualization ({max_count} Samples/Class) - {split.capitalize()} Set"
        )

        # Save t-SNE plot as PNG
        tsne_dir = os.path.join(args.analysis_path, "tsne")
        os.makedirs(tsne_dir, exist_ok=True)
        fig_path = os.path.join(tsne_dir, f"{split}_tsne.png")
        fig.write_image(fig_path)
        logging.info(f"Saved t-SNE plot for {split} set at {fig_path}")

    # Return paths for train/test t-SNE plots
    return { "train": "/analysis/tsne/train_tsne.png", "test": "/analysis/tsne/test_tsne.png" }
