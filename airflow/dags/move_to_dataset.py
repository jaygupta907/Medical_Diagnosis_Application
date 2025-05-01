import sqlite3
import os
import io
from PIL import Image
import subprocess
import requests




def move_to_dataset(db_path="/opt/airflow/uploads/predictions.db", datasets_root="/opt/airflow/datasets/chestxray/Data/train"):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Fetch all predictions
        cursor.execute("SELECT id, image_data, predicted_label, corrected_label FROM prediction ORDER BY id ASC;")
        rows = cursor.fetchall()

        if not rows:
            print("No prediction records found.")
            return

        for row in rows:
            _, image_data, predicted_label, corrected_label = row
            label = corrected_label if corrected_label else predicted_label
            label = label.upper()
            class_dir = os.path.join(datasets_root, label)
            os.makedirs(class_dir, exist_ok=True)
            print(class_dir)

            # Count existing images in the class folder
            existing_images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            image_index = len(existing_images)
            image_filename = f"{label}({image_index}).jpg"
            image_path = os.path.join(class_dir, image_filename)

            try:
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                image.save(image_path)
            except Exception as e:
                print(f"Failed to save image to {image_path}: {e}")
                continue

            # Delete the record after saving the imagedoc
            # cursor.execute("DELETE FROM prediction WHERE id = ?", (row[0],))

        conn.commit()
        print(f"Moved {len(rows)} images to dataset folders.")

    except Exception as e:
        print(f"Error while moving predictions to dataset: {e}")

    finally:
        cursor.close()
        conn.close()
