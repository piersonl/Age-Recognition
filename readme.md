# Age Group Recognition from Face Images (CNN + SQLite)

This project implements an end-to-end **age group recognition system** that:

1. Trains a **Convolutional Neural Network (CNN)** to classify faces into  
   **YOUNG**, **MIDDLE**, or **OLD**.
2. Stores images, labels, and predictions in a **SQLite database**.
3. Provides a **user-friendly Python API** to:
   - Predict ages for new images or folders of images.
   - Automatically store predictions and confidence scores in the database.
   - Query and inspect prediction statistics.

---

## 🎯 Goal of the Project

The main goal is to build a **complete age recognition pipeline** that goes beyond a standalone model:

- **Model side:** Train and evaluate a CNN to classify face images into three age groups.
- **System side:** Integrate the model with a SQLite database and helper scripts so that:
  - A user can predict the age group of a person in an image with a single function call.
  - The image, prediction, and metadata are automatically recorded and can be queried later.

In other words:  
> **User supplies image(s) → CNN predicts age group → database stores everything (image, prediction, confidence, timestamps).**

---

## 📁 Repository Structure

Key files:

- `Age-Recognition-CNN.ipynb`  
  - Builds, trains, and saves the CNN model.  
  - Shows evaluation metrics and confusion matrix.  
  - Demonstrates the high-level workflow with `workflow_manager.py`.

- `make_db.py`  
  - Creates the SQLite database and the `images` table.  
  - Loads **training images** (with known labels YOUNG/MIDDLE/OLD) and general metadata.

- `predict_and_update.py`  
  - Helper functions for using the **saved CNN model** to predict ages.  
  - Handles:
    - A **single file** path  
    - A **folder of images**  
    - **All images already in the DB** that don’t yet have a predicted age  
  - After predicting, updates the database with `predicted_age`, `confidence`, and timestamps.

- `workflow_manager.py`  
  - Wraps the lower-level helper functions into a **clean, user-facing API**.  
  - Exposes functions like:
    - `predict(path)`
    - `predict_all_pending()`
    - `show_stats()`
    - `query_predictions(...)`

- `training_data.db` (or similar)  
  - SQLite database file created by `make_db.py`.

- `confusion_matrix.png` (optional, but referenced below)  
  - Visualization of the test-set confusion matrix.

---

## 📦 Requirements / Dependencies

You can adapt this to your environment, but a typical setup uses:

- **Python 3.10+** (or similar)
- Core libraries:
  - `tensorflow` / `keras`
  - `numpy`
  - `pandas`
  - `opencv-python`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
- Database:
  - `sqlite3` (standard Python library, no extra install needed)
- Optional:
  - `jupyter` for running the notebook

Example installation (inside a virtual environment is recommended):

```bash
pip install tensorflow numpy pandas opencv-python matplotlib seaborn scikit-learn jupyter