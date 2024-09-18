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

# 1. MATRIZ DE COVARIANZA
X = df[['GDP', 'Population']].values
X_meaned = X - np.mean(X, axis=0)
cov_mat = np.cov(X_meaned, rowvar=False)

# 2. EIGENVALUES Y EIGENVECTORS
eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)
eigenvalues = eigenvalues[::-1] 
eigenvectors = eigenvectors[:, ::-1]

# 3. VARIANZA EXPLICADA POR EL EIGENVALUE
explained_variance_ratio = eigenvalues / np.sum(eigenvalues)

# 4. EIGENVECTOR (primer componente)
eigenvector1 = eigenvectors[:, 0]

# 5. CIUDADES EN 1 DIMENSIÓN
X_pca = X_meaned.dot(eigenvector1)

# Agregar la componente principal al DataFrame
df['Principal_Component'] = X_pca

# Ordenar las ciudades por la primera componente
sorted_cities = df[['City', 'Principal_Component']].sort_values(by='Principal_Component', ascending=False).head(10)

# Mostrar resultados
print("============= PCA ===============\n")
print("1. Matriz de Covarianza:\n", cov_mat, "\n")
print("2. Valores Propios (Eigenvalues):", eigenvalues, "\n")
print("Vectores Propios (Eigenvectors):\n", eigenvectors, "\n")
print("3. Varianza Explicada por el Eigenvalue:", explained_variance_ratio[0], "\n")
print("4. Valor del Eigenvector:\n", eigenvector1, "\n")
print("5. Primeras 10 ciudades ordenadas por la componente principal:\n", sorted_cities, "\n")

