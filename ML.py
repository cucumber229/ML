import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 1. Загрузка и предобработка
data = pd.read_csv('Files/AmesHousing.csv')

# Оставляем только численные признаки
numeric = data.select_dtypes(include='number')
# Удаляем сильно коррелирующие признаки (|corr| > 0.81)
corr = numeric.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.81)]
clean = numeric.drop(columns=to_drop).dropna()

# Нормализация
scaler = StandardScaler()
scaled = pd.DataFrame(scaler.fit_transform(clean), columns=clean.columns)

# 2. 3D-график через PCA
features = scaled.drop('SalePrice', axis=1)
target = scaled['SalePrice']

pca = PCA(n_components=2)
reduced = pca.fit_transform(features)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection='3d')
ax.scatter(reduced[:,0], reduced[:,1], target, c=target)
ax.set_xlabel('PCA-1')
ax.set_ylabel('PCA-2')
ax.set_zlabel('SalePrice (scaled)')
plt.show()

# 3. Разбиение на train/test
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# 4. Подбор alpha и оценка RMSE на тесте
alphas = np.logspace(-4, 2, 50)
rmse_test = []

for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_test)
    rmse_test.append(np.sqrt(mean_squared_error(y_test, y_pred)))

# График зависимости ошибки от alpha
plt.figure(figsize=(8, 5))
plt.semilogx(alphas, rmse_test)
plt.xlabel('Alpha (коэффициент регуляризации)')
plt.ylabel('RMSE на тесте')
plt.title('Зависимость RMSE от alpha')
plt.show()

# 5. Лучший alpha и важный признак
best_idx = np.argmin(rmse_test)
best_alpha = alphas[best_idx]
best_rmse = rmse_test[best_idx]

best_lasso = Lasso(alpha=best_alpha, max_iter=10000)
best_lasso.fit(X_train, y_train)

coef_abs = np.abs(best_lasso.coef_)
most_imp = features.columns[np.argmax(coef_abs)]

print(f"Лучший alpha: {best_alpha:.4f}")
print(f"RMSE на тесте при лучшем alpha: {best_rmse:.4f}")
print(f"Наиболее влиятельный признак: {most_imp}")
