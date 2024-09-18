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
train_cities = ["Caracas", "Maracaibo", "Rio de Janeiro", "Santiago", "Lima", 
                "Bogotá", "Monterrey", "Brasília", "Medellín", 
                "Curitiba", "Santo Domingo", "Panama City", "San Salvador", 
                "Santa Cruz", "Montevideo", "Salvador", "Fortaleza", 
                "Asunción", "Managua", "Rosario", "Córdoba", "São Paulo"]

test_cities = ["Mexico City", "Buenos Aires", "Quito", "Guadalajara", 
               "Porto Alegre", "Recife", "Belo Horizonte", "Havana", "La Paz"]

# Filtrar datos de entrenamiento y prueba
train_df = df[df['City'].isin(train_cities)]
test_df = df[df['City'].isin(test_cities)]

# Definir X y Y para entrenamiento
X_train = train_df['Population'].values
y_train = train_df['GDP'].values

# a) Calcular b1 (pendiente) y b0 (intersección)
n = len(X_train)
mean_x = np.mean(X_train)
mean_y = np.mean(y_train)

# Numerador y denominador para b1
numerador = np.sum((X_train - mean_x) * (y_train - mean_y))
denominador = np.sum((X_train - mean_x) ** 2)
b1 = numerador / denominador

# b0 (intercepto)
b0 = mean_y - b1 * mean_x

# Ecuación de la regresión
print(f"Ecuación de la regresión: Y = {b0:.2f} + {b1:.2f}X")

# b) cálculo de MSE y MAPE
y_train_pred = b0 + b1 * X_train
y_test_pred = b0 + b1 * test_df['Population'].values

# Calcular MSE
mse_train = np.mean((y_train - y_train_pred) ** 2)
mse_test = np.mean((test_df['GDP'].values - y_test_pred) ** 2)

# Calcular MAPE
mape_train = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
mape_test = np.mean(np.abs((test_df['GDP'].values - y_test_pred) / test_df['GDP'].values)) * 100

# c) Cálculo de R², RSS y TSS
rss_train = np.sum((y_train - y_train_pred) ** 2)
tss_train = np.sum((y_train - mean_y) ** 2)
r2_train = 1 - (rss_train / tss_train)

rss_test = np.sum((test_df['GDP'].values - y_test_pred) ** 2)
tss_test = np.sum((test_df['GDP'].values - np.mean(test_df['GDP'].values)) ** 2)
r2_test = 1 - (rss_test / tss_test)

# Mostrar resultados
print("\nResultados:")
print(f"MSE (Train): {mse_train:.2f}")
print(f"MSE (Test): {mse_test:.2f}")
print(f"MAPE (Train): {mape_train:.2f}%")
print(f"MAPE (Test): {mape_test:.2f}%")
print(f"R² (Train): {r2_train:.2f}")
print(f"RSS (Train): {rss_train:.2f}")
print(f"TSS (Train): {tss_train:.2f}")
print(f"R² (Test): {r2_test:.2f}")
print(f"RSS (Test): {rss_test:.2f}")
print(f"TSS (Test): {tss_test:.2f}")
