import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse as argparse


def create_database(file_path: str):
    """Creates a pandas DataFrame from the given file path.
    Args:
        file_path (str): The path to the file (CSV).
    Returns:
        pd.DataFrame: The resulting DataFrame."""

    try:
        temp_db = pd.read_csv(file_path)
        return (get_column_names(temp_db))
    except FileNotFoundError as f:
        print(f"FileNotFoundError: {f}") #!
    except ValueError as v:
        print(f"ValueError: {v}") #!


def get_column_names(temp_db: pd.DataFrame):
    """"""
    temp_db.columns = ["id", "diagnosis",
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
        "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
        "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
        "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"]
    temp_db["diagnosis"] = temp_db["diagnosis"].map({"B": 0, "M":1})
    return (temp_db)


def cor_matrix(db: pd.DataFrame, show: bool) -> pd.DataFrame:
    """"""
    corr = db.corr()
    if show:
        plt.figure(figsize=(10, 8))
        sns.heatmap(db.corr(), annot=False, cmap='coolwarm')
        plt.show()

    filtered_cols = []
    for col, value in corr['diagnosis'].items():
        if (col == 'diagnosis' or col == 'id'):
            continue
        if (abs(value) < 0.3):
            filtered_cols.append(col)
    formated_db = db.drop(filtered_cols, axis=1)
    return formated_db


def separate_dataset(dataset: pd.DataFrame, seed: int, train_ratio: float = 0.8):
    """"""
    dataset = dataset.sample(frac=1, random_state=seed)
    split_index = int(len(dataset) * train_ratio)
    train_dt = dataset.iloc[:split_index]
    validation_dt = dataset.iloc[split_index:]
    return (train_dt, validation_dt)


def save_to_csv(dataset: pd.DataFrame, filename: str):
    """"""
    dataset.to_csv(filename)
    print(f"Dataset saved in {filename}")


def main():
    """"""
    parser = argparse.ArgumentParser(description="Help Separate mode", add_help=False)
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='Show this help message and quit')
    parser.add_argument('--dataset', type=str, help='Path to Dataset')
    parser.add_argument('--mode', type=str, choices=['graph', 'separate'], default='train', help='Execution Mode')
    parser.add_argument('--seed', type=int, default=1349, help='Seed to separate dataset')
    args = parser.parse_args()
    try:
        assert args.dataset == "data/data.csv", "Wrong path to dataset"
        db = create_database(args.dataset)
        if (args.mode == "graph"):
            dataset = cor_matrix(db, True)
        elif (args.mode == "separate"):
            dataset = cor_matrix(db, False)
            train_dt, validation_dt = separate_dataset(dataset, args.seed)
            save_to_csv(train_dt, "~/Documents/multilayer_perceptron/src/data/train.csv")
            save_to_csv(validation_dt, "~/Documents/multilayer_perceptron/src/data/val.csv")
    except AssertionError as e:
        print(f"AssertionError: {e}")


if __name__ == "__main__":
    main()