import sys as sys
import argparse as argparse
import pandas as pd
from split_data import create_database


def csv_to_pandas(filepath: str) -> pd.DataFrame:
    """Creates a pandas DataFrame from the given file path.
    Args:
        file_path (str): The path to the file (CSV).
    Returns:
        pd.DataFrame: The resulting DataFrame."""

    try:
        temp_db = pd.read_csv(filepath)
        return (temp_db)
    except FileNotFoundError as f:
        print(f"FileNotFoundError: {f}") #!
    except ValueError as v:
        print(f"ValueError: {v}") #!
    

def train():
    """"""


def main():
    parser = argparse.ArgumentParser(description="Help Train mode", add_help=False)
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='Show this help message and quit')
    parser.add_argument('--layer', type=str, default='24 24 24', help='Number of Layer')
    parser.add_argument('--epochs', type=int, default='84', help='Number of time the model train itself')
    parser.add_argument('--loss', type=str, default='categoricalCrossentropy', help='Methods of calculate loss')
    parser.add_argument('--batch_size', type=int, default='8', help='???')
    parser.add_argument('--learning_rate', type=float, default='0.0314', help='Speed of adjustments for the model')
    parser.add_argument('--seed', type=int, default='1349', help='Seed for weights and bias initialisation')
    args = parser.parse_args()
    print(args)

    dt_train = csv_to_pandas("data/train_dataset.csv")
    dt_valid = csv_to_pandas("data/validation_dataset.csv")


if __name__ == "__main__":
    main()
