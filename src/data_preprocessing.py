import pandas as pd
from sklearn.model_selection import train_test_split
import os

def preprocess_data():
    # Path to the dataset
    data_path = os.path.join(os.path.dirname(__file__), '../data/nba_games.csv')

    # Load the dataset
    data = pd.read_csv(data_path)

    # Example: Selecting relevant features
    features = ['fg%', '3p%', 'ft', 'tov%', 'ortg_max', 'drtg_max', 'home']
    target = 'won'

    # Handle missing values
    data = data[features + [target]].dropna()

    # Encode the target (True -> 1, False -> 0)
    data[target] = data[target].astype(int)

    # Split into training and testing sets
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save processed data for training
    X_train.to_csv('../data/X_train.csv', index=False)
    X_test.to_csv('../data/X_test.csv', index=False)
    y_train.to_csv('../data/y_train.csv', index=False)
    y_test.to_csv('../data/y_test.csv', index=False)

    print("Data preprocessing completed and saved!")

if __name__ == "__main__":
    preprocess_data()