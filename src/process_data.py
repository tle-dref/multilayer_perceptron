import pandas as pd
import sys as sys
import matplotlib as plt
import seaborn as sbn
import os as os
import pickle as plk
from sklearn.model_selection import train_test_split

DATA_PATH = "data/processed_data.plk"

def save_processed_data(X_train, X_val, y_train, y_val, mean, std):
    with open(DATA_PATH, 'wb') as f:
        plk.dump({
            'X_train': X_train,
            'X_val': X_val,
            'y_train': y_train,
            'y_val': y_val,
            'mean': mean,
            'std': std
        }, f)

def load_processed_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Processed data not found. Please run with --data first.")
    with open(DATA_PATH, 'rb') as f:
        return plk.load(f)

def init_data(file_path):
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

def plot_feature_distribution(data, feature):
    malignant = data[data['diagnosis'] == 1][feature]
    benign = data[data['diagnosis'] == 0][feature]

    plt.hist([benign, malignant], bins=30, label=['Benign', 'Malignant'], color=['lightblue', 'salmon'], alpha=0.7)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_feature_boxplot(data, feature):
    benign = data[data['diagnosis'] == 0][feature]
    malignant = data[data['diagnosis'] == 1][feature]

    plt.boxplot([benign, malignant], labels=['Benign', 'Malignant'])
    plt.title(f'{feature} by Diagnosis')
    plt.ylabel(feature)
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(data):
    corr_matrix = data.corr()

    plt.figure(figsize=(12, 10))
    plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title('Feature Correlation Matrix')

    ticks = range(len(corr_matrix.columns))
    plt.xticks(ticks, corr_matrix.columns, rotation=90, fontsize=6)
    plt.yticks(ticks, corr_matrix.columns, fontsize=6)

    plt.tight_layout()
    plt.show()

def visualize_data(data):
    plot_class_distribution(data)
    feature = [col for col in data.columns if col != 'diagnosis']
    #for column in feature:
        #plot_feature_boxplot(data, column)
        #plot_feature_distribution(data, column)
    #this shows us that symmetry_mean,fractal_dimension_mean,texture_se, smoothness_se, symmetry_se
    #are not very usefull for our model to train on
    plot_correlation_matrix(data)
    corr_with_target = data.corr()['diagnosis'].drop('diagnosis').sort_values(ascending=False)
    print(corr_with_target)
    selected_feature = [feat for feat, corr in data.corr()['diagnosis'].drop('diagnosis').items() if abs(corr) >= 0.4]
    print(40*'-' + "selected_feature" + 40 * '-')
    print(selected_feature)
    return selected_feature
