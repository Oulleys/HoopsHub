from src.data_preprocessing import preprocess_data
from src.train_model import train
from src.predict import make_predictions

if __name__ == "__main__":
    # Step 1: Preprocess the data
    preprocess_data()

    # Step 2: Train the model
    train()

    # Step 3: Make predictions
    make_predictions()