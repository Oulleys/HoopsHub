import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

def make_predictions():
    # Load the processed test data
    X_test = pd.read_csv('../data/X_test.csv')
    y_test = pd.read_csv('../data/y_test.csv')

    # Load the trained model
    model_path = '../models/nba_model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate performance
    accuracy = (predictions == y_test.values.ravel()).mean()
    print(f"Model Accuracy: {accuracy:.2f}")

    # Save predictions for analysis
    X_test['Predicted_Win'] = predictions
    X_test['Actual_Win'] = y_test.values.ravel()
    predictions_file_path = '../data/predictions.csv'
    X_test.to_csv(predictions_file_path, index=False)
    print(f"Predictions saved to {predictions_file_path}!")

if __name__ == "__main__":
    make_predictions()