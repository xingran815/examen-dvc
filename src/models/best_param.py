import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


def train_model():
    # Get the directory of the current script
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))

    # Define paths
    PROCESSED_DATA_DIR = os.path.join(THIS_DIR, "../../data/processed_data")
    MODELS_DIR = os.path.join(THIS_DIR, "../../models")

    # Input files
    X_train_path = os.path.join(PROCESSED_DATA_DIR, "X_train_scale.csv")
    y_train_path = os.path.join(PROCESSED_DATA_DIR, "y_train.csv")

    # Output file
    output_file = os.path.join(MODELS_DIR, "best_params.pkl")

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

    # Initialize model
    rf = RandomForestRegressor(random_state=42)

    # Define parameters to test
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_leaf': [1, 2, 4]
    }

    # Perform Grid Search
    print("Starting Grid Search...")
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        scoring='neg_mean_squared_error',
        verbose=3
    )

    grid_search.fit(X_train, y_train)

    # Get best parameters
    best_params = grid_search.best_params_
    print(f"Best parameters found: {best_params}")

    # Save best parameters to .pkl file
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(best_params, f)

    print(f"Best parameters saved to {output_file}")


if __name__ == "__main__":
    train_model()
