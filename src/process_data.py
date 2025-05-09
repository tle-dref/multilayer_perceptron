import pandas as pd
import sys as sys
# import matplotlib.pyplot as plt

#! --> Error tested !

def create_database(file_path: str):
    """Creates a pandas DataFrame from the given file path.
    Args:
        file_path (str): The path to the file (CSV).
    Returns:
        pd.DataFrame: The resulting DataFrame."""

    try:
        temp_db = pd.read_csv(file_path)
        return (get_column_names(temp_db))
    except FileNotFoundError:
        print(f"Error: The file at path '{file_path}' was not found.") #!
    except pd.errors.EmptyDataError:
        print(f"Error: The file at path '{file_path}' is empty.") #!


def get_column_names(temp_db: pd.DataFrame):
    temp_db.columns = ["id", "diagnosis",
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
        "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
        "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
        "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"]
    return (temp_db)


def main():
    try:
        assert len(sys.argv) == 2, "path to dataset is required (\"../data/data.csv\")" #!
        file_path = sys.argv[1]
        db = create_database(file_path)
        if (db is None):
            return
    except AssertionError as e:
        print(f"{e}")


if __name__ == "__main__":
    main()
