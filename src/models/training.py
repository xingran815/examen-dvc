import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestRegressor


def training():
    # Get the directory of the current script
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))

    # Define paths
    PROCESSED_DATA_DIR = os.path.join(THIS_DIR, "../../data/processed_data")
    MODELS_DIR = os.path.join(THIS_DIR, "../../models")

    # Input files
    X_train_path = os.path.join(PROCESSED_DATA_DIR, "X_train_scale.csv")
    y_train_path = os.path.join(PROCESSED_DATA_DIR, "y_train.csv")

    # Parameter file
    param_file = os.path.join(MODELS_DIR, "best_params.pkl")

    # Output file
    output_file = os.path.join(MODELS_DIR, "model.pkl")

    # Check if data exists
    if not os.path.exists(X_train_path) or not os.path.exists(y_train_path):
        print(f"Error: Training data not found in {PROCESSED_DATA_DIR}.")
        print("Please run 'src/data/data_splitting.py' and \
            'src/data/data_normalize.py' first.")
        return

    # Load data
    print("Loading data...")
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)

    # Flatten y_train to 1D array (required for sklearn)
    y_train = y_train.values.ravel()

    # Load best parameters
    print("Loading best parameters...")
    # Check if parameter file exists
    if not os.path.exists(param_file):
        print(f"Error: Parameter file not found in {MODELS_DIR}.")
        print("Please run 'src/models/best_param.py' first.")
        return
    with open(param_file, 'rb') as f:
        best_params = pickle.load(f)
    print(f"Best parameters found: {best_params}")

    # Train the model
    model = RandomForestRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)

    # Save the model
    with open(output_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {output_file}")


if __name__ == "__main__":
    training()
