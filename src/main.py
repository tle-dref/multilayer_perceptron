import pandas as pd
import sys as sys
import os as os
import pickle as plk
import matplotlib as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from process_data import init_data, visualize_data, load_processed_data, save_processed_data
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
        selected_feature = visualize_data(data)
        X = data[selected_feature]
        y = data['diagnosis']
        # Split the data into training and validation sets, important to prevent overfitting
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.2,       # 80% train, 20% validation
            random_state=42,     # for reproducibility
            stratify=y           # keeps class balance (same amount of benign/malignant in train/val)
        )
        # this is the part where we scale our data, wich puts all the data in the same range, mean is 0 and std is 1
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
        layers_config = [(20, 'sigmoid'), (10, 'sigmoid'), (1, 'sigmoid')]
        model = MLP(layers_config)

        model.fit(X_train, y_train)
        print("Model training done.")
    else:
        print(usage)

if __name__ == "__main__":
    main()
