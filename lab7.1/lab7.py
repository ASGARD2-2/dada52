import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#завантаження датасету
df = pd.read_csv("lab7.1/diabetes_data.csv")

print("Колонки у файлі:")
print(df.columns)
print("\nПерші 5 рядків:")
print(df.head())

#визначення цільової колонки
target_col = "Diagnosis"

if target_col not in df.columns:
    raise ValueError("У датасеті немає колонки Diagnosis!")

#видалення несуттєвих колонок
drop_cols = ["PatientID", "DoctorInCharge"]
for col in drop_cols:
    if col in df.columns:
        df = df.drop(columns=[col])

#заповнення пропусків
for col in df.columns:
    if col != target_col:
        df[col] = df[col].fillna(df[col].median())

#розділення на X та y
X = df.drop(columns=[target_col])
y = df[target_col]

#масштабування ознак
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#розбиття train test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

#результат
print("\n  дані")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
