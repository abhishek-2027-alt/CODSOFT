import zipfile
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def unzip_dataset():
    zip_path = r'C:\Users\abhis\OneDrive\Desktop\Data science\titanic.zip'  # Absolute path to ZIP file
    extract_folder = 'titanic_data'

    if not os.path.exists(extract_folder):
        os.makedirs(extract_folder)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)

    print(f"[INFO] Dataset extracted to folder: {extract_folder}")

def load_data():
    print("[INFO] Loading Titanic dataset...")

    dataset_path = 'titanic_data/Titanic-Dataset (1).csv'  # Correct file inside the extracted folder
    df = pd.read_csv(dataset_path)

    print(f"[INFO] Dataset shape: {df.shape}")
    return df

def preprocess_data(df):
    print("[INFO] Preprocessing data...")

    # Drop columns that are not useful
    columns_to_drop = ['Name', 'Ticket', 'Cabin', 'PassengerId']
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(col, axis=1)

    # Handle missing values
    if 'Age' in df.columns:
        df['Age'].fillna(df['Age'].median(), inplace=True)
    if 'Embarked' in df.columns:
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    # Encode categorical variables
    label_encoder = LabelEncoder()
    if 'Sex' in df.columns:
        df['Sex'] = label_encoder.fit_transform(df['Sex'])
    if 'Embarked' in df.columns:
        df['Embarked'] = label_encoder.fit_transform(df['Embarked'])

    return df

def split_data(df):
    print("[INFO] Splitting data into train and test sets...")
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    print("[INFO] Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    print("[INFO] Evaluating model...")
    y_pred = model.predict(X_test)

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Plot Feature Importances
    feature_importances = pd.Series(model.feature_importances_, index=X_test.columns)
    feature_importances.sort_values().plot(kind='barh', figsize=(10,6))
    plt.title('Feature Importances')
    plt.show()

def main():
    unzip_dataset()
    df = load_data()
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

if __name__ == '__main__':
    main()
