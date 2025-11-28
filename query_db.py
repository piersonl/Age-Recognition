import sqlite3
import os
import csv
from PIL import Image
import io

# This file queries and exports our images + labels from training_data.db for our CNN to interpret

def export_images_and_csv(db_path, output_image_dir, output_csv_path):
    # Ensure output folder exists
    os.makedirs(output_image_dir, exist_ok=True)

    # Connect to database
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Query all records
    cur.execute("SELECT id, class, image FROM images")
    rows = cur.fetchall()

    print(f"[INFO] Found {len(rows)} images in database.")

    # Prepare CSV
    with open(output_csv_path, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ID", "Class"])   # CSV header

        # Iterate records
        for img_id, class_label, blob_data in rows:
            # Reconstruct image from BLOB
            image = Image.open(io.BytesIO(blob_data)).convert("RGB")

            # Save image as ID.jpg
            image_path = os.path.join(output_image_dir, f"{img_id}.jpg")
            image.save(image_path)

            # Write CSV line
            writer.writerow([img_id, class_label])

            print(f"[OK] Exported {image_path}  |  Class={class_label}")

    conn.close()
    print(f"\n[SUCCESS] Export complete.")
    print(f"Images saved to: {output_image_dir}")
    print(f"CSV rebuilt at:   {output_csv_path}")


# --- Example Usage ---
if __name__ == "__main__":
    export_images_and_csv(
        db_path="training_data.db",
        output_image_dir="reconstructed_images",
        output_csv_path="reconstructed_labels.csv"
    )