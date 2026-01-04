from sklearn.preprocessing import StandardScaler
import os
import pandas as pd


def data_normalize():
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_FILE = os.path.join(THIS_DIR,
                              "../../data/processed_data/X_train.csv")
    X_train = pd.read_csv(INPUT_FILE)
    INPUT_FILE = os.path.join(THIS_DIR,
                              "../../data/processed_data/X_test.csv")
    X_test = pd.read_csv(INPUT_FILE)

    # normalize the X_train and X_test data
    print("Normalize the training and testing input data...")
    scaler = StandardScaler()
    X_train_scale = pd.DataFrame(scaler.fit_transform(X_train),
                                 columns=X_train.columns)
    X_test_scale = pd.DataFrame(scaler.transform(X_test),
                                columns=X_test.columns)

    # output the normalized data
    print("Saving the normalized data...")
    OUTPUT_FILE = os.path.join(THIS_DIR,
                               "../../data/processed_data/X_train_scale.csv")
    X_train_scale.to_csv(OUTPUT_FILE, index=False)
    OUTPUT_FILE = os.path.join(THIS_DIR,
                               "../../data/processed_data/X_test_scale.csv")
    X_test_scale.to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    data_normalize()
