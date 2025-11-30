import sqlite3
import os
import csv
from PIL import Image
import io
from datetime import datetime

def create_database(db_path, image_folder=None, csv_path=None):
    """
    Creates training_data.db with enhanced schema for predictions.
    Age categories: YOUNG, MIDDLE, OLD
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Enhanced table with prediction fields
    cur.execute("""
        CREATE TABLE IF NOT EXISTS images(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            class TEXT CHECK(class IN ('YOUNG', 'MIDDLE', 'OLD')),
            predicted_age TEXT CHECK(predicted_age IN ('YOUNG', 'MIDDLE', 'OLD')),
            confidence REAL,
            image BLOB NOT NULL,
            is_training BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            predicted_at TIMESTAMP
        )
    """)
    
    print("[INFO] Enhanced 'images' table created.")
    print("[INFO] Age categories: YOUNG, MIDDLE, OLD")
    
    # if initial training data provided, load it
    if image_folder and csv_path:
        load_training_data(conn, image_folder, csv_path)
    
    conn.commit()
    conn.close()
    print(f"[SUCCESS] Database ready: {db_path}")


def load_training_data(conn, image_folder, csv_path):
    """Load initial training images with known labels."""
    cur = conn.cursor()
    
    # read labels from CSV
    labels = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # adjust column names to match your CSV
            filename = row.get('ID')
            age_class = row.get('Class')
            
            # validate age class
            if age_class.upper() not in ['YOUNG', 'MIDDLE', 'OLD']:
                print(f"[WARNING] Invalid age class '{age_class}' for {filename}, skipping...")
                continue
                
            labels[filename] = age_class.upper()
    
    print(f"[INFO] Loading {len(labels)} training images...")
    
    count = 0
    for filename, class_label in labels.items():
        image_path = os.path.join(image_folder, filename)
        
        if not os.path.exists(image_path):
            print(f"[SKIP] Not found: {image_path}")
            continue
        
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG')
                img_blob = img_byte_arr.getvalue()
                
                cur.execute("""
                    INSERT INTO images (filename, class, image, is_training) 
                    VALUES (?, ?, ?, 1)
                """, (filename, class_label, img_blob))
                
                count += 1
                if count % 50 == 0:
                    print(f"[PROGRESS] Loaded {count} images...")
                    
        except Exception as e:
            print(f"[ERROR] {filename}: {e}")
    
    conn.commit()
    print(f"[SUCCESS] Loaded {count} training images.")
    
    # show distribution
    cur.execute("SELECT class, COUNT(*) FROM images WHERE is_training=1 GROUP BY class")
    distribution = cur.fetchall()
    print("\n[INFO] Training data distribution:")
    for age_class, count in distribution:
        print(f"  {age_class}: {count} images")


def add_new_image(db_path, image_path, filename=None):
    """
    Add a new image to database for prediction.
    Returns the new image ID.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    if filename is None:
        filename = os.path.basename(image_path)
    
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            img_blob = img_byte_arr.getvalue()
            
            cur.execute("""
                INSERT INTO images (filename, image, is_training) 
                VALUES (?, ?, 0)
            """, (filename, img_blob))
            
            image_id = cur.lastrowid
            conn.commit()
            conn.close()
            
            print(f"[OK] Added image ID {image_id}: {filename}")
            return image_id
            
    except Exception as e:
        print(f"[ERROR] Failed to add image: {e}")
        conn.close()
        return None


def view_database_stats(db_path):
    """View statistics about the database."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # total images
    cur.execute("SELECT COUNT(*) FROM images")
    total = cur.fetchone()[0]
    
    # training vs prediction images
    cur.execute("SELECT COUNT(*) FROM images WHERE is_training=1")
    training = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM images WHERE is_training=0")
    prediction = cur.fetchone()[0]
    
    # predicted vs unpredicted
    cur.execute("SELECT COUNT(*) FROM images WHERE predicted_age IS NOT NULL")
    predicted = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM images WHERE predicted_age IS NULL AND is_training=0")
    unpredicted = cur.fetchone()[0]
    
    # distribution by class
    cur.execute("""
        SELECT predicted_age, COUNT(*) 
        FROM images 
        WHERE predicted_age IS NOT NULL 
        GROUP BY predicted_age
    """)
    pred_distribution = cur.fetchall()
    
    conn.close()
    
    print("\n" + "="*50)
    print("DATABASE STATISTICS")
    print("="*50)
    print(f"Total images: {total}")
    print(f"  Training images: {training}")
    print(f"  Prediction images: {prediction}")
    print(f"\nPrediction status:")
    print(f"  Predicted: {predicted}")
    print(f"  Awaiting prediction: {unpredicted}")
    
    if pred_distribution:
        print(f"\nPredicted age distribution:")
        for age_class, count in pred_distribution:
            print(f"  {age_class}: {count} images")
    print("="*50 + "\n")


if __name__ == "__main__":
    # create database with training data
    create_database(
        db_path="faces.db",
        image_folder="faces\\Train",  
        csv_path="faces\\train.csv"            
    )
    # View stats
    view_database_stats("faces.db")