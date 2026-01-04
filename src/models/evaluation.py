import pandas as pd
import os
import pickle
import json
from sklearn.metrics import mean_squared_error, r2_score


def predict_and_evaluate():
    # Get the directory of the current script
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))

    # Define paths
    PROCESSED_DATA_DIR = os.path.join(THIS_DIR, "../../data/processed_data")
    MODELS_DIR = os.path.join(THIS_DIR, "../../models")
    METRICS_DIR = os.path.join(THIS_DIR, "../../metrics")

    # Input files
    X_test_path = os.path.join(PROCESSED_DATA_DIR, "X_test_scale.csv")
    y_test_path = os.path.join(PROCESSED_DATA_DIR, "y_test.csv")

    # Output file
    output_prediction_file = os.path.join(PROCESSED_DATA_DIR, "y_predict.csv")
    output_evaluation_file = os.path.join(METRICS_DIR, "scores.json")
    # Model file
    model_file = os.path.join(MODELS_DIR, "model.pkl")

    # Check if data exists
    if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
        print(f"Error: Test data not found in {PROCESSED_DATA_DIR}.")
        print("Please run 'src/data/data_splitting.py' and \
            'src/data/data_normalize.py' first.")
        return

    # Load data
    print("Loading data...")
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)
    # flattern y_test to 1D array
    y_test = y_test.values.ravel()

    # Check if the model exists
    if not os.path.exists(model_file):
        print(f"Error: Model file not found in {MODELS_DIR}.")
        print("Please run 'src/models/model.py' first.")
        return

    # Load model
    print("Loading model...")
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)

    # Evaluate the model
    print("Evaluating model...")
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    metrics = {
        "mse": mse,
        "r2": r2
    }
    print(f"Evaluation metrics: {metrics}")

    # Save predictions
    print(f"Saving predictions to {output_prediction_file}...")
    y_pred_df = pd.DataFrame(y_pred, columns=['silica_concentrate'])
    y_pred_df.to_csv(output_prediction_file, index=False)

    # Save evaluation metrics
    print(f"Saving evaluation metrics to {output_evaluation_file}...")
    with open(output_evaluation_file, 'w') as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    predict_and_evaluate()
