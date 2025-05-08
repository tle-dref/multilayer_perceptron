import pandas as pd
import sys as sys
# import matplotlib.pyplot as plt

def create_database(file_path: str):
    """Creates a pandas DataFrame from the given file path.
    Args:
        file_path (str): The path to the file (CSV).
    Returns:
        pd.DataFrame: The resulting DataFrame."""

    try:
        temp_db = pd.read_csv(file_path)
        return (temp_db)
    except FileNotFoundError:
        print(f"Error: The file at path '{file_path}' was not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: The file at path '{file_path}' is empty.")


def main():
    try:
        assert len(sys.argv) == 2, "path to dataset is required (\"../data/data.csv\")"
        file_path = sys.argv[1]
        db = create_database(file_path)
        print(db)
    except AssertionError as e:
        print(f"{e}")


if __name__ == "__main__":
    main()
