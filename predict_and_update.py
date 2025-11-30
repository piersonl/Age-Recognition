import sqlite3
import io
import os
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import numpy as np
from datetime import datetime

# age categories for your model
CLASSES = ['YOUNG', 'MIDDLE', 'OLD']


def load_model(model_path):
    """Load the trained CNN model and return model with its expected input shape."""
    try:
        model = keras.models.load_model(model_path)
        
        # get the expected input shape from the model
        input_shape = model.input_shape
        # input_shape is (None, height, width, channels)
        expected_size = (input_shape[1], input_shape[2])
        
        print(f"[INFO] Model loaded from: {model_path}")
        print(f"[INFO] Model expects input size: {expected_size}")
        return model, expected_size
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None, None


def preprocess_image(image, target_size):
    """Preprocess image for model input."""
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0  # normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array


def predict_age(model, image, target_size):
    """
    Run prediction on a single image.
    Returns predicted class (YOUNG/MIDDLE/OLD) and confidence.
    """
    img_array = preprocess_image(image, target_size)
    predictions = model.predict(img_array, verbose=0)
    
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])
    predicted_age = CLASSES[predicted_class_idx]
    
    return predicted_age, confidence, predictions[0]


def predict_from_path(path, model_path, db_path="training_data.db", add_to_db=True):
    """
    Universal prediction function - handles both single images and folders.
    
    Args:
        path: Path to either an image file or a folder of images
        model_path: Path to the trained model
        db_path: Database path (for storing results)
        add_to_db: Whether to add images and predictions to database
    
    Returns:
        Dictionary with results summary
    """
    if not os.path.exists(path):
        print(f"[ERROR] Path not found: {path}")
        return None
    
    # Load model once and get expected input size
    model, target_size = load_model(model_path)
    if model is None:
        return None
    
    # Determine if it's a file or folder
    if os.path.isfile(path):
        # Single image
        print(f"\n{'='*60}")
        print(f"PREDICTING SINGLE IMAGE")
        print(f"{'='*60}\n")
        result = _predict_single_image(path, model, target_size, db_path, add_to_db)
        return result
    
    elif os.path.isdir(path):
        # Folder of images
        print(f"\n{'='*60}")
        print(f"BATCH PREDICTION FROM FOLDER: {path}")
        print(f"{'='*60}\n")
        result = _predict_folder(path, model, target_size, db_path, add_to_db)
        return result
    
    else:
        print(f"[ERROR] Invalid path: {path}")
        return None


def _predict_single_image(image_path, model, target_size, db_path, add_to_db):
    """Internal function to predict a single image."""
    try:
        # load and predict on image
        image = Image.open(image_path).convert("RGB")
        predicted_age, confidence, all_probs = predict_age(model, image, target_size)
        
        # display results
        filename = os.path.basename(image_path)
        print(f"IMAGE: {filename}")
        print(f"{'-'*60}")
        print(f"Predicted Age: {predicted_age}")
        print(f"Confidence: {confidence:.1%}")
        print(f"\nProbability Distribution:")
        for cls, prob in zip(CLASSES, all_probs):
            bar = '█' * int(prob * 40)
            print(f"  {cls:8s}: {prob:6.1%} {bar}")
        print(f"{'='*60}\n")
        
        # optionally add to database
        db_id = None
        if add_to_db and db_path:
            db_id = _save_to_database(image, filename, predicted_age, confidence, db_path)
        
        return {
            'success': True,
            'filename': filename,
            'predicted_age': predicted_age,
            'confidence': confidence,
            'probabilities': dict(zip(CLASSES, all_probs)),
            'db_id': db_id
        }
        
    except Exception as e:
        print(f"[ERROR] Failed to process {image_path}: {e}")
        return {'success': False, 'error': str(e)}


def _predict_folder(folder_path, model, target_size, db_path, add_to_db):
    """Internal function to predict all images in a folder."""
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(image_extensions)]
    
    if len(image_files) == 0:
        print(f"[INFO] No image files found in {folder_path}")
        return {'success': False, 'error': 'No images found'}
    
    print(f"[INFO] Found {len(image_files)} images\n")
    
    # track results
    results_summary = {'YOUNG': 0, 'MIDDLE': 0, 'OLD': 0}
    successful = 0
    failed = 0
    all_results = []
    
    # process each image
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        
        try:
            # load and predict
            image = Image.open(img_path).convert("RGB")
            predicted_age, confidence, all_probs = predict_age(model, image, target_size)
            
            results_summary[predicted_age] += 1
            successful += 1
            
            # Display result
            prob_str = " | ".join([f"{cls}: {prob:.1%}" for cls, prob in zip(CLASSES, all_probs)])
            print(f"[OK] {img_file}")
            print(f"     Predicted: {predicted_age} (confidence: {confidence:.1%})")
            print(f"     Probabilities: {prob_str}")
            
            # Save to database
            db_id = None
            if add_to_db and db_path:
                db_id = _save_to_database(image, img_file, predicted_age, confidence, db_path)
                if db_id:
                    print(f"     Database ID: {db_id}")
            
            print("-"*60)
            
            all_results.append({
                'filename': img_file,
                'predicted_age': predicted_age,
                'confidence': confidence,
                'db_id': db_id
            })
            
        except Exception as e:
            print(f"[ERROR] {img_file}: {e}")
            print("-"*60)
            failed += 1
    
    # print summary
    print(f"\n{'='*60}")
    print("BATCH PREDICTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total images: {len(image_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nAge Distribution:")
    for age_class in CLASSES:
        count = results_summary[age_class]
        percentage = (count / successful * 100) if successful > 0 else 0
        print(f"  {age_class}: {count} images ({percentage:.1f}%)")
    print(f"{'='*60}\n")
    
    return {
        'success': True,
        'total': len(image_files),
        'successful': successful,
        'failed': failed,
        'distribution': results_summary,
        'results': all_results
    }


def _save_to_database(image, filename, predicted_age, confidence, db_path):
    """Save image and prediction to database."""
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        # convert image to BLOB
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_blob = img_byte_arr.getvalue()
        
        # insert with prediction
        cur.execute("""
            INSERT INTO images (filename, image, predicted_age, confidence, is_training, predicted_at) 
            VALUES (?, ?, ?, ?, 0, ?)
        """, (filename, img_blob, predicted_age, confidence, datetime.now()))
        
        image_id = cur.lastrowid
        conn.commit()
        conn.close()
        
        return image_id
        
    except Exception as e:
        print(f"     [WARNING] Could not save to database: {e}")
        return None


def predict_unpredicted_images(db_path, model_path):
    """
    Find all images in database without predictions and run them through the model.
    Updates database with predictions.
    """
    model, target_size = load_model(model_path)
    if model is None:
        return
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # get images that haven't been predicted yet
    cur.execute("""
        SELECT id, image, filename 
        FROM images 
        WHERE predicted_age IS NULL AND is_training = 0
    """)
    rows = cur.fetchall()
    
    if len(rows) == 0:
        print("[INFO] No unpredicted images found in database.")
        conn.close()
        return
    
    print(f"\n{'='*60}")
    print(f"PREDICTING DATABASE IMAGES")
    print(f"{'='*60}")
    print(f"[INFO] Found {len(rows)} unpredicted images\n")
    
    predictions_summary = {'YOUNG': 0, 'MIDDLE': 0, 'OLD': 0}
    
    for img_id, blob_data, filename in rows:
        try:
            # load image from BLOB
            image = Image.open(io.BytesIO(blob_data)).convert("RGB")
            
            # Predict
            predicted_age, confidence, all_probs = predict_age(model, image, target_size)
            predictions_summary[predicted_age] += 1
            
            # Update database
            cur.execute("""
                UPDATE images 
                SET predicted_age = ?, 
                    confidence = ?,
                    predicted_at = ?
                WHERE id = ?
            """, (predicted_age, confidence, datetime.now(), img_id))
            
            # display prediction
            prob_str = " | ".join([f"{cls}: {prob:.1%}" for cls, prob in zip(CLASSES, all_probs)])
            print(f"[OK] ID {img_id} ({filename or 'unknown'})")
            print(f"     Predicted: {predicted_age} (confidence: {confidence:.1%})")
            print(f"     Probabilities: {prob_str}")
            print("-"*60)
            
        except Exception as e:
            print(f"[ERROR] ID {img_id}: {e}")
    
    conn.commit()
    conn.close()
    
    print(f"\n{'='*60}")
    print("PREDICTION SUMMARY")
    print(f"{'='*60}")
    for age_class in CLASSES:
        count = predictions_summary[age_class]
        print(f"{age_class}: {count} images")
    print(f"{'='*60}\n")
    print(f"[SUCCESS] Database predictions complete!")