import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

def train():
    # Load processed training data
    X_train = pd.read_csv('../data/X_train.csv')
    y_train = pd.read_csv('../data/y_train.csv')

    # Train a Random Forest Classifier
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train.values.ravel())

    # Save the trained model
    model_path = '../models/nba_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model training completed and saved to {model_path}!")

if __name__ == "__main__":
    train()