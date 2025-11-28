import os
import sqlite3
import csv

# This file builds our db, no need to run unless training_data.db needs to be rebuilt

def create_database(db_path="images.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY,
            class TEXT NOT NULL,
            image BLOB NOT NULL
        );
    """)

    conn.commit()
    conn.close()
    print(f"[INFO] Database created: {db_path}")


def image_to_blob(image_path):
    """
    Reads an image file and converts it to a BLOB for SQLite.
    """
    with open(image_path, "rb") as f:
        return f.read()


def insert_record(db_path, img_id, class_label, image_bytes):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO images (id, class, image) VALUES (?, ?, ?)",
        (img_id, class_label, image_bytes)
    )

    conn.commit()
    conn.close()


def build_database(image_folder, csv_path, db_path="images.db"):
    # Create DB and table
    create_database(db_path)

    # Load CSV
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"[INFO] Loaded {len(rows)} CSV entries.")

    # Insert each record
    for row in rows:
        raw_id = row["ID"].strip()

        # Remove any extension (.jpg, .png, etc.)
        if "." in raw_id:
            raw_id = raw_id.split(".")[0]

        img_id = int(raw_id)
        class_label = row["Class"]

        image_filename = f"{img_id}.jpg"
        image_path = os.path.join(image_folder, image_filename)

        if not os.path.exists(image_path):
            print(f"[WARNING] Image missing: {image_path} — skipping.")
            continue

        image_bytes = image_to_blob(image_path)
        insert_record(db_path, img_id, class_label, image_bytes)

        print(f"[INFO] Inserted ID={img_id}, Class={class_label}")

    print(f"\n[SUCCESS] Database '{db_path}' built successfully.")


if __name__ == "__main__":
    # Path to our folder with training images
    imgpath = 'local/train'
    # Path to our label map
    csvpath = 'local/train.csv'
    # Name of the db we will create
    dbname = 'training_data.db'

    build_database(imgpath, csvpath, dbname)