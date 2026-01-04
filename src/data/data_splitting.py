from sklearn.model_selection import train_test_split
import pandas as pd
import os


def split_data():
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_FILE = os.path.join(THIS_DIR, "../../data/raw_data/raw.csv")
    df = pd.read_csv(INPUT_FILE)

    # drop Date column
    df = df.drop(columns=['date'])

    y = df["silica_concentrate"]
    X = df.drop(columns=["silica_concentrate"])

    # split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7
    )

    OUTPUT_FILE = os.path.join(THIS_DIR,
                               "../../data/processed_data/X_test.csv")
    X_test.to_csv(OUTPUT_FILE, index=False)
    OUTPUT_FILE = os.path.join(THIS_DIR,
                               "../../data/processed_data/y_test.csv")
    y_test.to_csv(OUTPUT_FILE, index=False)
    OUTPUT_FILE = os.path.join(THIS_DIR,
                               "../../data/processed_data/X_train.csv")
    X_train.to_csv(OUTPUT_FILE, index=False)
    OUTPUT_FILE = os.path.join(THIS_DIR,
                               "../../data/processed_data/y_train.csv")
    y_train.to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    split_data()
