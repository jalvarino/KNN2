import pandas as pd
import numpy as np
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score

# =========================================================
# Paso 1: Cargar la base de datos
# =========================================================
df = pd.read_csv(r"C:\Temp-univ\KNN2\cleaned_dataset.csv")

print("Paso 1 ✅ Dataset cargado")
print("Shape:", df.shape)
print("Columnas:", df.columns.tolist())

# Columnas
TARGET = "Outcome"
FEATURES = [c for c in df.columns if c != TARGET]


# =========================================================
# Paso 2: Subconjuntos aleatorios: 20 train + 20 test (40 total)
# =========================================================

df_40 = df.sample(n=40, random_state = 1).reset_index(drop=True)
df_train_20 = df_40.iloc[:20].copy()
df_test_20  = df_40.iloc[20:].copy()

print("\nPaso 2 ✅ Subconjuntos aleatorios creados")
print("Train20:", df_train_20.shape, "Test20:", df_test_20.shape)


# =========================================================
# Paso 3: Implementar distancia euclidiana + prueba con ejemplo
# =========================================================
def distancia_euclidiana(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.sqrt(np.sum((x - y) ** 2)))

# Ejemplo
f1 = [1, 106, 70, 28, 135, 34.2, 0.142, 22]  # Fila 1
f2 = [2, 102, 86, 36, 120, 45.5, 0.127, 23]  # Fila 2

diff = np.array(f1, dtype=float) - np.array(f2, dtype=float)
suma_cuadrados = np.sum(diff ** 2)
dist_manual = np.sqrt(suma_cuadrados)
dist_func = distancia_euclidiana(f1, f2)

print("\nPaso 3 ✅ Distancia Euclidiana (verificación)")
print("Diferencias:", diff)
print("Suma de cuadrados:", suma_cuadrados)
print("Distancia (manual):", dist_manual)
print("Distancia (función):", dist_func)


# =========================================================
# Paso 4: Implementar KNN básico con k=3
#       - Distancia a todos los puntos de entrenamiento
#       - Elegir 3 vecinos y votar mayoría
#       - Aplicar a 10 test usando 10 train
# =========================================================
def knn_predecir_uno(X_train, y_train, x_test, k=3):
    distancias = []
    for i in range(len(X_train)):
        d = distancia_euclidiana(X_train[i], x_test)
        distancias.append((d, y_train[i]))

    distancias.sort(key=lambda t: t[0])
    vecinos = distancias[:k]
    clases = [clase for (_, clase) in vecinos]
    return Counter(clases).most_common(1)[0][0]

def knn_predecir_batch(X_train, y_train, X_test, k=3):
    return np.array([knn_predecir_uno(X_train, y_train, x, k=k) for x in X_test])

# Usar 10 train y 10 test
X_train_10 = df_train_20[FEATURES].iloc[:10].to_numpy()
y_train_10 = df_train_20[TARGET].iloc[:10].to_numpy()

X_test_10 = df_test_20[FEATURES].iloc[:10].to_numpy()
y_test_10 = df_test_20[TARGET].iloc[:10].to_numpy()

y_pred_10 = knn_predecir_batch(X_train_10, y_train_10, X_test_10, k=3)

tabla_knn_basico = pd.DataFrame({
    "Real (Outcome)": y_test_10,
    "Predicho (KNN básico k=3)": y_pred_10
})

print("\nPaso 4 ✅ Tabla (10 test vs 10 train) usando KNN básico")
print(tabla_knn_basico)


# =========================================================
# Paso 5: Usar toda la data 80/20 con estratificación
# =========================================================
X = df[FEATURES].to_numpy()
y = df[TARGET].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nPaso 5 ✅ Split 80/20 estratificado")
print("Train:", X_train.shape, "Test:", X_test.shape)
print("Proporción clase 1 train:", y_train.mean(), " | test:", y_test.mean())


# =========================================================
# Paso 6: KNN con datos crudos (sin escalar) + accuracy
# =========================================================
k = 3
knn_raw = KNeighborsClassifier(n_neighbors=k, metric="minkowski", p=2)  # Euclidiana
knn_raw.fit(X_train, y_train)
pred_raw = knn_raw.predict(X_test)
acc_raw = accuracy_score(y_test, pred_raw)

print("\nPaso 6 ✅ Accuracy sin escalar:", acc_raw)


# =========================================================
# Paso 7: Min-Max Scaling + KNN + accuracy
# =========================================================
mm = MinMaxScaler()
X_train_mm = mm.fit_transform(X_train)  # fit solo en train
X_test_mm = mm.transform(X_test)

knn_mm = KNeighborsClassifier(n_neighbors=k, metric="minkowski", p=2)
knn_mm.fit(X_train_mm, y_train)
pred_mm = knn_mm.predict(X_test_mm)
acc_mm = accuracy_score(y_test, pred_mm)

print("Paso 7 ✅ Accuracy Min-Max:", acc_mm)


# =========================================================
# Paso 8: Z-score + KNN + accuracy
# =========================================================
sc = StandardScaler()
X_train_z = sc.fit_transform(X_train)   # fit solo en train
X_test_z = sc.transform(X_test)

knn_z = KNeighborsClassifier(n_neighbors=k, metric="minkowski", p=2)
knn_z.fit(X_train_z, y_train)
pred_z = knn_z.predict(X_test_z)
acc_z = accuracy_score(y_test, pred_z)

print("Paso 8 ✅ Accuracy Z-score:", acc_z)


# =========================================================
# Paso 10/11: Tabla comparativa de accuracies
# =========================================================
tabla_comparativa = pd.DataFrame({
    "Experimento": [
        "KNN sin escalar (80/20)",
        "KNN normalizado Min-Max (80/20)",
        "KNN estandarizado Z-score (80/20)"
    ],
    "Accuracy": [acc_raw, acc_mm, acc_z]
})

print("\n===============================")
print("Paso 10/11 ✅ Tabla comparativa")
print("===============================")
print(tabla_comparativa)