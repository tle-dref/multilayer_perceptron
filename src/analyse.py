import sys as sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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


def cor_matrix(db: pd.DataFrame, show: bool):
    """"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(db.corr(),annot=False, cmap='coolwarm')
    tab = db.corr().values
    # [print(f"{x > 0.3}") for x in tab]
    # print(tab.values)
    plt.show()


def main():
    """"""


if __name__ == "__main__":
    main()