from numpy import select
import pandas as pd
import sys as sys
import os as os
import pickle as plk
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from process_data import init_data, visualize_data, load_processed_data, save_processed_data, select_features
from mlp import MLP


def main():
    usage = "Usage:\n  --data <path_to_the_data>\n  --train \n --predict"
    if len(sys.argv) < 2:
        print(usage)
        return
    if sys.argv[1] == "--data":
        if len(sys.argv) < 3:
            print("Please provide a path to the data file.")
            return
        file = sys.argv[2]
        data = init_data(file)
        visualize_data(data)
        selected_feature = select_features(data)
        with open('selected_features.pkl', 'wb') as f:
            plk.dump(selected_feature, f)
        X = data[selected_feature]
        y = data['diagnosis']
        # Split the data into training and validation sets, important to prevent overfitting
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.2,       # 80% train, 20% validation
            random_state=42,     # for reproducibility
            stratify=y           # keeps class balance (same amount of benign/malignant in train/val)
        )
        # this is the part where we scale our data, which puts all the data in the same range, mean is 0 and std is 1
        mean = X_train.mean()
        std = X_train.std()
        X_train = (X_train - mean) / std
        #use the same mean and std to scale the validation set to prevent data leakage and misleading results
        X_val = (X_val - mean) / std
        save_processed_data(X_train, X_val, y_train, y_val, mean, std)
        print("Data processed and saved. Now run with --train to train the model.")

    elif sys.argv[1] == "--train":
        try:
            data = load_processed_data()
        except FileNotFoundError as e:
            print(e)
            return
        X_train = data['X_train']
        y_train = data['y_train']
        X_val = data['X_val']
        y_val = data['y_val']

        # Define model architecture
        input_size = X_train.shape[1]  # Number of features
        layers_config = [(input_size, 'sigmoid'), (10, 'sigmoid'), (1, 'sigmoid')]

        # Create and train model with validation data
        model = MLP(layers_config, learning_rate=0.01)
        history = model.fit(
            X_train, y_train,
            x_val=X_val, y_val=y_val,
            epochs=100,
            batch_size=32,
            early_stopping=True,
            patience=10
        )
        # Save the trained model
        with open('model.pkl', 'wb') as f:
            plk.dump(model, f)
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Training History')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        # Evaluate final model performance
        final_val_loss, final_val_acc = model.evaluate(X_val, y_val)
        print(f"Final validation accuracy: {final_val_acc:.4f}")
        print("Model training done.")

    elif sys.argv[1] == "--predict":
        if len(sys.argv) < 3:
            print("Please provide a path to the data file for prediction.")
            return
        # Load the trained model
        try:
            with open('model.pkl', 'rb') as f:
                model = plk.load(f)
        except FileNotFoundError:
            print("Model not found. Please train the model first.")
            return
        # Load the data statistics for preprocessing
        try:
            data = load_processed_data()
            mean = data['mean']
            std = data['std']
            with open('selected_features.pkl', 'rb') as f:
                selected_feature = plk.load(f)
        except (FileNotFoundError, KeyError):
            print("Preprocessing data not found. Please process data first.")
            return
        # Load and preprocess new data
        file = sys.argv[2]
        try:
            new_data = init_data(file)
            y_val = new_data['diagnosis']
            X_new = new_data[selected_feature]
            X_new = (X_new - mean) / std
            predictions = model.predict(X_new)
            predictions_proba = model.predict_proba(X_new)
            new_data['probability'] = predictions_proba
            new_data['predicted'] = predictions

            print("Predictions saved to 'predictions.csv'")
            new_data = new_data[['diagnosis', 'predicted', 'probability']]
            y_proba = new_data['probability'].values.flatten()
            y_val = new_data['diagnosis'].values.flatten()
            loss = model._loss(y_val, y_proba)
            print(f"Binary Cross-Entropy Loss: {loss:.4f}")
            new_data.to_csv('predictions.csv', index=False)
        except Exception as e:
            print(f"Error during prediction: {e}")

    else:
        print(usage)

if __name__ == "__main__":
    main()
