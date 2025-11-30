from make_db import view_database_stats
from predict_and_update import predict_from_path, predict_unpredicted_images
import sqlite3

# Configuration
DB_PATH = "faces.db"
MODEL_PATH = "age_classifier_model.h5"


def predict(path, save_to_db=True):
    """
    Universal prediction function - automatically handles files or folders.
    
    Args:
        path: Path to either an image file or a folder of images
        save_to_db: Whether to save images and predictions to database
    
    Returns:
        Dictionary with results
    """
    return predict_from_path(path, MODEL_PATH, DB_PATH, add_to_db=save_to_db)


def predict_all_pending():
    """Predict all images in the database that don't have predictions yet."""
    predict_unpredicted_images(DB_PATH, MODEL_PATH)


def show_stats():
    """Display database statistics."""
    view_database_stats(DB_PATH)


def query_predictions(age_category=None, min_confidence=None):
    """
    Query and display predictions from the database.
    
    Args:
        age_category: Filter by age category ('YOUNG', 'MIDDLE', 'OLD')
        min_confidence: Minimum confidence threshold (0.0 to 1.0)
    
    Returns:
        List of prediction results
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    query = "SELECT id, filename, predicted_age, confidence, predicted_at FROM images WHERE predicted_age IS NOT NULL"
    params = []
    
    if age_category:
        query += " AND predicted_age = ?"
        params.append(age_category.upper())
    
    if min_confidence:
        query += " AND confidence >= ?"
        params.append(min_confidence)
    
    query += " ORDER BY confidence DESC"
    
    cur.execute(query, params)
    results = cur.fetchall()
    conn.close()
    
    if not results:
        print("[INFO] No predictions found matching criteria.")
        return []
    
    print(f"\n{'='*60}")
    print(f"PREDICTION RESULTS ({len(results)} images)")
    if age_category:
        print(f"Filtered by: {age_category}")
    if min_confidence:
        print(f"Minimum confidence: {min_confidence:.1%}")
    print(f"{'='*60}\n")
    
    for img_id, filename, pred_age, confidence, pred_time in results:
        print(f"ID {img_id}: {filename or 'unknown'}")
        print(f"  Age: {pred_age} | Confidence: {confidence:.1%} | Predicted: {pred_time}")
        print("-"*60)
    
    return results