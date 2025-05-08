import pandas as pd
import sys as sys
import matplotlib as plt
import seaborn as sbn

def process_data(file_path):
    data = pd.read_csv(file_path, header=None)

    columns = [
        'id', 'diagnosis',
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
        'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se',
        'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
        'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]

    data.columns = columns
    data = data.drop(columns=['id'])
    data['diagnosis'] = data["diagnosis"].map({'M': 1, 'B': 0})
    print(data.head)

    print(60*"-")
    print(data.isnull().sum())

    return data


import matplotlib.pyplot as plt

def plot_class_distribution(data):
    counts = data['diagnosis'].value_counts()
    labels = ['Benign', 'Malignant']
    values = [counts[0], counts[1]]
    plt.bar(labels, values, color=['lightblue', 'salmon'])
    plt.title('Diagnosis Class Distribution')
    plt.ylabel('Count')

    for i, val in enumerate(values):
        plt.text(i, val + 5, str(val), ha='center')

    plt.tight_layout()
    plt.show()

def visualize_data(data):
    plot_class_distribution(data)
def main():
    file = sys.argv[1]
    data = process_data(file)
    visualize_data(data)

if __name__ == "__main__":
    main()
