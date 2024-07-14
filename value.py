import numpy as np
import matplotlib.pyplot as plt

# Datos de entrada
X = np.array([[0, 1], [2, 0], [1, 1]])
# Añadir una columna de unos para el término de bias
X = np.hstack((np.ones((X.shape[0], 1)), X))
# Clases correspondientes
y = np.array([1, 1, -1])

# Inicializar pesos
weights = np.zeros(X.shape[1])
learning_rate = 1
epochs = 25

# Función de activación
def step_function(z):
    return 1 if z >= 0 else -1

# Entrenamiento del Perceptrón
for epoch in range(epochs):
    for i in range(len(X)):
        # Producto escalar entre pesos y datos de entrada
        linear_output = np.dot(X[i], weights)
        y_pred = step_function(linear_output)
        
        # Actualización de pesos
        if y_pred != y[i]:
            weights += learning_rate * y[i] * X[i]

print("Pesos finales:", weights)

# Representación gráfica
plt.figure(figsize=(8, 6))

# Dibujar los puntos
for i in range(len(X)):
    if y[i] == 1:
        plt.scatter(X[i][1], X[i][2], marker='o', color='r')
    else:
        plt.scatter(X[i][1], X[i][2], marker='x', color='b')

# Dibujar el hiperplano
x_values = np.linspace(-1, 3, 100)
y_values = -(weights[1] * x_values + weights[0]) / weights[2]
plt.plot(x_values, y_values, label='Hiperplano')

plt.xlabel('X1')
plt.ylabel('X2')
plt.xlim(-1, 3)
plt.ylim(-1, 3)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.legend()
plt.title('Representación del Perceptrón y los datos')
plt.show()
