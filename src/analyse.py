import sys as sys
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from process_data import *

#! --> Error tested !

def random_forest(db: pd.DataFrame):
    # Sample dataset
    X = db.drop(['id', 'diagnosis'], axis=1)
    y = db['diagnosis']

    # Fit model
    model = RandomForestClassifier(n_estimators=100, random_state=12)
    model.fit(X, y)

    # Feature importances
    importances = model.feature_importances_
    features = X.columns

    # Plot
    plt.figure(figsize=(8, 4))
    plt.barh(features, importances)
    plt.xlabel("Feature Importance")
    plt.title("Random Forest Feature Ranking")
    plt.show()


def bar_chart(db: pd.DataFrame):
    m_row = db[db["diagnosis"] == 1].iloc[0, 2:]
    b_row = db[db["diagnosis"] == 0].iloc[0, 2:]
    plt.figure(figsize=(8, 5))
    plt.bar(m_row.index, m_row.values, color="red")
    plt.bar(b_row.index, b_row.values, color="mediumseagreen")
    plt.xticks(rotation=90)
    plt.title('Data malignant or benign')
    plt.tight_layout()
    plt.show()


def main():
    if (len(sys.argv) != 3):
        print("Wrong number of args --> 2 or 3") #!
        return()
    file_path = sys.argv[1]
    db = create_database(file_path)
    if (db is None):
        return()
    if (len(sys.argv) == 3 and sys.argv[2] == "--rdmforest"):
        random_forest(db)
    elif (len(sys.argv) == 3 and sys.argv[2] == "--barchart"):
        bar_chart(db)
    else:
        print(f"Flag '{sys.argv[2]}' not found. ex: --rdmforest, --barchart") #!


if __name__ == "__main__":
    main()