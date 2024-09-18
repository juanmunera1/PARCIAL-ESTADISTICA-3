import numpy as np
import pandas as pd

# Datos de las ciudades
data = {
    "City": ["Caracas", "Maracaibo", "Rio de Janeiro", "Santiago", "Lima", 
             "Bogotá", "Monterrey", "Buenos Aires", "Brasília", "Medellín", 
             "Curitiba", "Santo Domingo", "Panama City", "San Salvador", 
             "Santa Cruz", "Montevideo", "Salvador", "Fortaleza", 
             "Asunción", "Managua", "Quito", "Guadalajara", 
             "Porto Alegre", "Recife", "Belo Horizonte", "Havana", 
             "La Paz", "Rosario", "Córdoba", "São Paulo", "Mexico City"],
    "Population": [2.9, 1.6, 6.7, 6.8, 10.7, 7.4, 4.9, 15, 4.7, 2.5,
                   3.5, 3.4, 1.5, 1.1, 1.5, 1.8, 2.9, 3.9, 0.7, 1,
                   2.8, 5.2, 4.3, 1.6, 2.7, 2.2, 0.8, 1.3, 1.5,
                   12.4, 9.2],
    "GDP": [30, 14, 172, 166, 94, 95, 93, 112, 66, 40,
            44, 30, 60, 20, 12, 45, 42, 34, 12, 8,
            26, 50, 55, 26, 57, 15, 10, 18, 21,
            337, 411]
}

# Crear DataFrame
df = pd.DataFrame(data)

# Definir conjuntos de entrenamiento y prueba
training_cities = ["Caracas", "Maracaibo", "Rio de Janeiro", "Santiago", "Lima", 
                   "Bogotá", "Monterrey", "Brasília", "Medellín", 
                   "Curitiba", "Santo Domingo", "Panama City", "San Salvador", 
                   "Santa Cruz", "Montevideo", "Salvador", "Fortaleza", 
                   "Asunción", "Managua", "Rosario", "Córdoba", "São Paulo"]

testing_cities = ["Mexico City", "Buenos Aires", "Quito", "Guadalajara", 
                  "Porto Alegre", "Recife", "Belo Horizonte", "Havana", "La Paz"]

# Filtrar datos de entrenamiento y prueba
train_data = df[df['City'].isin(training_cities)]
test_data = df[df['City'].isin(testing_cities)]

# Definir variables X (Población) e y (PIB) para entrenamiento y prueba
X_train = train_data[['Population']].values
y_train = train_data['GDP'].values
X_test = test_data[['Population']].values
y_test = test_data['GDP'].values

def DISTANCIA(point1, point2):
    """Calcula la distancia euclidiana entre dos puntos."""
    return np.sqrt(np.sum((point1 - point2) ** 2))

def KNN(X_train, y_train, X_test, k):
    """Realiza la predicción utilizando el algoritmo K-Nearest Neighbors."""
    predictions = []
    for test_point in X_test:
        # Calcular distancias desde el punto de prueba a todos los puntos de entrenamiento
        distances = [DISTANCIA(test_point, train_point) for train_point in X_train]
        # Obtener los índices de los K vecinos más cercanos
        k_nearest_indices = np.argsort(distances)[:k]
        # Obtener los valores de los K vecinos más cercanos
        k_nearest_values = [y_train[i] for i in k_nearest_indices]
        # Predicción como el promedio de los valores de los vecinos más cercanos
        predictions.append(np.mean(k_nearest_values))
    return np.array(predictions)

# Aplicar el modelo KNN con 3 vecinos
k = 3
y_train_predictions = KNN(X_train, y_train, X_train, k)
y_test_predictions = KNN(X_train, y_train, X_test, k)

# Calcular MSE (Error Cuadrático Medio)
mse_train = np.mean((y_train - y_train_predictions) ** 2)
mse_test = np.mean((y_test - y_test_predictions) ** 2)

# Calcular MAPE (Error Absoluto Porcentual Medio)
mape_train = np.mean(np.abs((y_train - y_train_predictions) / y_train)) * 100
mape_test = np.mean(np.abs((y_test - y_test_predictions) / y_test)) * 100

# Mostrar resultados
print("============= KNN  ===============\n")
print(f"MSE (Train): {mse_train:.2f}")
print(f"MSE (Test): {mse_test:.2f}")
print(f"MAPE (Train): {mape_train:.2f}%")
print(f"MAPE (Test): {mape_test:.2f}%")
