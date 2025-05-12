import pandas as pd
import sys as sys
import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestClassifier


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
    temp_db["diagnosis"] = temp_db["diagnosis"].map({"B": 0, "M":1})
    return (temp_db)


# def random_forest(db: pd.DataFrame):
#     # Sample dataset
#     X = db.drop(['id', 'diagnosis'], axis=1)
#     y = db['diagnosis']

#     # Fit model
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X, y)

#     # Feature importances
#     importances = model.feature_importances_
#     features = X.columns

#     # Plot
#     plt.figure(figsize=(8, 4))
#     plt.barh(features, importances)
#     plt.xlabel("Feature Importance")
#     plt.title("Random Forest Feature Ranking")
#     plt.show()


def bar_chart(db: pd.DataFrame):
    m_row = db[db["diagnosis"] == 'M'].iloc[0, 2:]
    b_row = db[db["diagnosis"] == 'B'].iloc[0, 2:]
    plt.figure(figsize=(8, 5))
    plt.bar(m_row.index, m_row.values, color="red")
    plt.bar(b_row.index, b_row.values, color="mediumseagreen")
    plt.xticks(rotation=90)
    plt.title('Data malignant or benign')
    plt.tight_layout()
    plt.show()


def main():
    try:
        assert len(sys.argv) == 2, "path to dataset is required (\"../data/data.csv\")" #!
        file_path = sys.argv[1]
        db = create_database(file_path)
        if (db is None):
            return
        # bar_chart(db)
        # random_forest(db)     #get useful features
    except AssertionError as e:
        print(f"{e}")


if __name__ == "__main__":
    main()
