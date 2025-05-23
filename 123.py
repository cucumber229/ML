import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import itertools
import imageio
import os

# Загружаем датасет ирисов
iris = datasets.load_iris()
X = iris.data
feature_names = iris.feature_names

# 1) Определяем оптимальное число кластеров (метод силуэта)
def choose_best_k(data, k_min=2, k_max=10):
    best_k, best_score = k_min, -1
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(data)
        score = silhouette_score(data, labels)
        print(f"k={k}, silhouette={score:.4f}")
        if score > best_score:
            best_k, best_score = k, score
    print(f"--> Выбранное число кластеров: {best_k}")
    return best_k

optimal_k = choose_best_k(X, 2, 6)

# 2) Своя реализация k-means

def init_centroids(data, k):
    indices = np.random.choice(len(data), size=k, replace=False)
    return data[indices].copy()


def assign_clusters(data, centroids):
    distances = np.linalg.norm(data[:, None] - centroids[None, :], axis=2)
    return np.argmin(distances, axis=1)


def update_centroids(data, labels, k):
    new_centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        pts = data[labels == i]
        new_centroids[i] = pts.mean(axis=0) if len(pts) > 0 else data[np.random.randint(len(data))]
    return new_centroids


def my_kmeans(data, k, max_iter=10, tol=1e-4, save_steps=True, out_dir="steps"):
    if save_steps:
        os.makedirs(out_dir, exist_ok=True)
    centroids = init_centroids(data, k)
    history = [centroids.copy()]
    for it in range(max_iter):
        labels = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, labels, k)
        history.append(new_centroids.copy())
        shift = np.linalg.norm(new_centroids - centroids)
        print(f"Iter {it+1}: centroid shift={shift:.4f}")
        centroids = new_centroids
        # Рисуем для первых двух признаков
        if save_steps:
            plt.figure(figsize=(6, 4))
            for cluster in range(k):
                plt.scatter(data[labels==cluster, 0], data[labels==cluster, 1], label=f"c{cluster}")
            plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, c='black')
            plt.title(f"Iteration {it+1}")
            plt.legend()
            plt.savefig(f"{out_dir}/step_{it+1:02d}.png")
            plt.close()
        if shift < tol:
            break
    return centroids, labels, history

centroids, labels, history = my_kmeans(X, optimal_k, max_iter=20)

# Собираем GIF из шагов
with imageio.get_writer('kmeans_iterations.gif', mode='I', duration=0.5) as writer:
    files = sorted([f for f in os.listdir('steps') if f.endswith('.png')])
    for fname in files:
        image = imageio.imread(os.path.join('steps', fname))
        writer.append_data(image)
print("Анимация шагов сохранена как kmeans_iterations.gif")

# 3) Финальный вывод всевозможных проекций
pairs = list(itertools.combinations(range(X.shape[1]), 2))
plt.figure(figsize=(12, 8))
for idx, (i, j) in enumerate(pairs, 1):
    plt.subplot(2, 3, idx)
    for cluster in range(optimal_k):
        pts = X[labels == cluster]
        plt.scatter(pts[:, i], pts[:, j], label=f"cl{cluster}")
    plt.xlabel(feature_names[i])
    plt.ylabel(feature_names[j])
    plt.xticks(rotation=45)
plt.suptitle('Все 2D проекции (финальный шаг)')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
