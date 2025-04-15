import os
import plotly.express as px
import logging
import argparse
from PIL import Image
import numpy as np
from sklearn.manifold import TSNE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser(description='EDA Arguments')
parser.add_argument('--dataset_path', type=str, default='datasets/chestxray/Data')
parser.add_argument('--analysis_path', type=str, default='analysis')
parser.add_argument('--image_size', type=int, default=128)
args = parser.parse_args()

os.makedirs(args.analysis_path, exist_ok=True)

def plot_distribution():
    for split in ['train', 'test']:
        split_path = os.path.join(args.dataset_path, split)
        
        if not os.path.isdir(split_path):
            logging.warning(f"{split_path} not found or is not a directory.")
            continue

        class_counts = {}
        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_path):
                image_count = len([
                    f for f in os.listdir(class_path)
                    if os.path.isfile(os.path.join(class_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ])
                class_counts[class_name] = image_count

        class_names = list(class_counts.keys())
        image_counts = list(class_counts.values())

        logging.info(f"{split.capitalize()} classes: {class_counts}")

        # Plotting
        fig = px.bar(
            x=class_names,
            y=image_counts,
            labels={'x': 'Class Name', 'y': 'Number of Images'},
            title=f'Image Distribution in {split.capitalize()} Set'
        )
        fig.update_layout(xaxis_tickangle=-45)

        distribution_path = os.path.join(args.analysis_path,"distribution")
        os.makedirs(distribution_path,exist_ok=True)
        png_path = os.path.join(distribution_path, f"{split}_image_distribution.png")


        fig.write_image(png_path)
        logging.info(f"Saved plots for {split} at {png_path}")
        return { "train": "/analysis/distribution/train_image_distribution.png", "test": "/analysis/distribution/test_image_distribution.png" }

def plot_tsne():
    for split in ['train', 'test']:
        split_path = os.path.join(args.dataset_path, split)
        if not os.path.isdir(split_path):
            logging.warning(f"{split_path} not found or is not a directory.")
            continue

        images = []
        labels = []
        class_image_counts = {}
        if split=='train':
            max_count = 400
        else:
            max_count = 100
        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if not os.path.isdir(class_path):
                continue

            count = 0
            for img_file in os.listdir(class_path):
                if count >=max_count:
                    break

                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, img_file)
                    try:
                        img = Image.open(img_path).convert("L").resize((args.image_size, args.image_size))
                        img_array = np.asarray(img).flatten() / 255.0
                        images.append(img_array)
                        labels.append(class_name)
                        count += 1
                    except Exception as e:
                        logging.warning(f"Failed to process {img_path}: {e}")

            class_image_counts[class_name] = count

        if not images:
            logging.warning(f"No images found in {split} for t-SNE.")
            continue

        logging.info(f"{split.capitalize()} samples used per class: {class_image_counts}")
        logging.info(f"Running t-SNE on {len(images)} total images from {split} set...")

        # 🔧 Convert list to NumPy array
        images_np = np.array(images)

        # 🔧 Replace deprecated `n_iter` with `max_iter`
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        tsne_result = tsne.fit_transform(images_np)

        fig = px.scatter(
            x=tsne_result[:, 0],
            y=tsne_result[:, 1],
            color=labels,
            labels={"x": "t-SNE Dimension 1", "y": "t-SNE Dimension 2"},
            title=f"t-SNE Visualization ({max_count} Samples/Class) - {split.capitalize()} Set"
        )

        tsne_dir = os.path.join(args.analysis_path, "tsne")
        os.makedirs(tsne_dir, exist_ok=True)
        fig_path = os.path.join(tsne_dir, f"{split}_tsne.png")
        fig.write_image(fig_path)
        logging.info(f"Saved t-SNE plot for {split} set at {fig_path}")
        return { "train": "/analysis/tsne/train_tsne.png", "test": "/analysis/tsne/test_tsne.png" }

# if __name__ == "__main__":
#     plot_distribution()
#     plot_tsne()