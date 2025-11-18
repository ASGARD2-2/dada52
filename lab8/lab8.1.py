import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns

#Завантаження даних
df = pd.read_csv("lab8\data.csv")

#Перевірка даних
print("Перші 5 рядків даних:")
print(df.head())
print(f"\nРозмірність даних: {df.shape}")
print(f"\nОсновні статистичні характеристики:")
print(df.describe())

#Візуалізація вихідних даних
plt.figure(figsize=(10, 6))
plt.scatter(df['x'], df['y'], alpha=0.6, s=30)
plt.title('Вихідні дані')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True, alpha=0.3)
plt.show()

#Визначення оптимальної кількості кластерів (метод ліктя)
X = df.values

#Масштабування даних
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Метод ліктя для визначення оптимального k
inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

#Графік методу ліктя
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, 'bo-')
plt.xlabel('Кількість кластерів (k)')
plt.ylabel('Inertia')
plt.title('Метод ліктя для визначення оптимального k')
plt.grid(True, alpha=0.3)
plt.show()

#Кластеризація з оптимальним k (з графіка видно, що оптимально k=4)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans.fit(X_scaled)

#Додаємо мітки кластерів до оригінального DataFrame
df['cluster'] = kmeans.labels_

#Розділення на окремі DataFrame для кожного кластера
cluster_dfs = {}
for cluster_id in range(optimal_k):
    cluster_dfs[f'cluster_{cluster_id}'] = df[df['cluster'] == cluster_id].copy()

#Вивід інформації про кластери
print("\n=== ІНФОРМАЦІЯ ПРО КЛАСТЕРИ ===")
for cluster_id in range(optimal_k):
    cluster_df = cluster_dfs[f'cluster_{cluster_id}']
    print(f"\nКластер {cluster_id}:")
    print(f"  Кількість точок: {len(cluster_df)}")
    print(f"  Центроїд: ({cluster_df['x'].mean():.2f}, {cluster_df['y'].mean():.2f})")
    print(f"  Діапазон x: [{cluster_df['x'].min()}, {cluster_df['x'].max()}]")
    print(f"  Діапазон y: [{cluster_df['y'].min()}, {cluster_df['y'].max()}]")

#Візуалізація результатів кластеризації
plt.figure(figsize=(12, 8))

#Кольори для кожного кластера
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

for cluster_id in range(optimal_k):
    cluster_df = cluster_dfs[f'cluster_{cluster_id}']
    plt.scatter(cluster_df['x'], cluster_df['y'], 
                c=colors[cluster_id], 
                label=f'Кластер {cluster_id} ({len(cluster_df)} точок)',
                alpha=0.7, 
                s=50)

#Відображення центроїдів
centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers_original[:, 0], centers_original[:, 1], 
            c='black', marker='X', s=200, label='Центроїди', alpha=0.8)

plt.title('Кластеризація k-means (k=4)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#Додаткова інформація про центроїди
print("\n=== ЦЕНТРОЇДИ КЛАСТЕРІВ ===")
for i, center in enumerate(centers_original):
    print(f"Кластер {i}: ({center[0]:.2f}, {center[1]:.2f})")

#Збереження кластерів в окремі CSV файли (опціонально)
for cluster_id in range(optimal_k):
    filename = f'cluster_{cluster_id}.csv'
    cluster_dfs[f'cluster_{cluster_id}'].to_csv(filename, index=False)
    print(f"Збережено: {filename}")

#Додатковий аналіз - матриця відстаней між центроїдами
from sklearn.metrics.pairwise import euclidean_distances

print("\n=== МАТРИЦЯ ВІДСТАНЕЙ МІЖ ЦЕНТРОЇДАМИ ===")
distance_matrix = euclidean_distances(centers_original)
distance_df = pd.DataFrame(distance_matrix, 
                          index=[f'Cluster_{i}' for i in range(optimal_k)],
                          columns=[f'Cluster_{i}' for i in range(optimal_k)])
print(distance_df.round(2))

#Статистика по кластерам
print("\n=== СТАТИСТИКА ПО КЛАСТЕРАМ ===")
cluster_stats = df.groupby('cluster').agg({
    'x': ['count', 'mean', 'std', 'min', 'max'],
    'y': ['mean', 'std', 'min', 'max']
}).round(2)

print(cluster_stats)

#Перевірка розподілу точок по кластерам
print(f"\nЗагальна кількість точок: {len(df)}")
print("Розподіл по кластерам:")
print(df['cluster'].value_counts().sort_index())