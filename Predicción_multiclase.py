import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
from my_module import dummy_class

# Definición de constantes
PLOT_COLORS = "ryb"
PLOT_STEP = 0.02

def decision_boundary(X, y, model, iris, two=None):
    """
    Función para trazar los límites de decisión del modelo.
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, PLOT_STEP),
                         np.arange(y_min, y_max, PLOT_STEP))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
    
    if two:
        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
        for i, color in zip(np.unique(y), PLOT_COLORS):
            idx = np.where(y == i)
            plt.scatter(X[idx, 0], X[idx, 1], label=y, cmap=plt.cm.RdYlBu, s=15)
        plt.show()
    else:
        set_ = {0, 1, 2}
        print(set_)
        for i, color in zip(range(3), PLOT_COLORS):
            idx = np.where(y == i)
            if np.any(idx):
                set_.remove(i)
                plt.scatter(X[idx, 0], X[idx, 1], label=y, cmap=plt.cm.RdYlBu, edgecolor='black', s=15)
        for i in set_:
            idx = np.where(iris.target == i)
            plt.scatter(X[idx, 0], X[idx, 1], marker='x', color='black')
        plt.show()

def plot_probability_array(X, probability_array):
    """
    Función para trazar la matriz de probabilidad.
    """
    plot_array = np.zeros((X.shape[0], 30))
    col_start = 0
    ones = np.ones((X.shape[0], 30))
    for class_, col_end in enumerate([10, 20, 30]):
        plot_array[:, col_start:col_end] = np.repeat(probability_array[:, class_].reshape(-1, 1), 10, axis=1)
        col_start = col_end
    plt.imshow(plot_array)
    plt.xticks([])
    plt.ylabel("samples")
    plt.xlabel("probability of 3 classes")
    plt.colorbar()
    plt.show()

# Cargamos el conjunto de datos iris
iris = datasets.load_iris()
pair = [0, 1]
X = iris.data[:, pair]
y = iris.target

# Visualizamos los datos
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.xlabel("sepal width (cm)")
plt.ylabel("petal width")

# Entrenamos un modelo de regresión logística
lr = LogisticRegression(random_state=0).fit(X, y)

# Calculamos las probabilidades predichas
probability = lr.predict_proba(X)

# Visualizamos la matriz de probabilidad
plot_probability_array(X, probability)

# Entrenamos un modelo SVM
model = SVC(kernel='linear', gamma=.5, probability=True)
model.fit(X, y)

# Hacemos predicciones con el modelo
yhat = model.predict(X)

# Calculamos la precisión del modelo
print("Accuracy: ", accuracy_score(y, yhat))

# Dibujamos los límites de decisión del modelo
decision_boundary(X, y, model, iris)

# Creamos una lista para almacenar los modelos
my_models = []

# Iteramos a través de cada clase
for class_ in np.unique(y):
    # Seleccionamos el índice de nuestra clase
    select = (y == class_)
    temp_y = np.zeros(y.shape)
    # Clase que estamos tratando de clasificar
    temp_y[y == class_] = class_
    # Establecemos otras muestras en una clase ficticia
    temp_y[y != class_] = dummy_class
    # Entrenamos el modelo y lo añadimos a la lista
    model = SVC(kernel='linear', gamma=.5, probability=True)
    my_models.append(model.fit(X, temp_y))
    # Dibujamos el límite de decisión
    decision_boundary(X, temp_y, model, iris)

# Creamos una matriz de probabilidad
probability_array = np.zeros((X.shape[0], 3))

# Iteramos a través de cada modelo
for j, model in enumerate(my_models):
    # Obtenemos la clase real
    real_class = np.where(np.array(model.classes_) != 3)[0]
    # Añadimos las probabilidades a la matriz
    probability_array[:, j] = model.predict_proba(X)[:, real_class][:, 0]

# Dibujamos la matriz de probabilidad
plot_probability_array(X, probability_array)

# Creamos un array para el voto mayoritario
majority_vote_array = np.zeros((X.shape[0], 3))
majority_vote_dict = {}

# Iteramos a través de cada modelo
for j, (model, pair) in enumerate(zip(my_models, pair)):
    majority_vote_dict[pair] = model.predict(X)
    majority_vote_array[:, j] = model.predict(X)

# Creamos un DataFrame con los votos mayoritarios
df_majority_vote = pd.DataFrame(majority_vote_dict)
print(df_majority_vote.head(10))

# Calculamos el voto mayoritario
one_vs_one = np.array([np.bincount(sample.astype(int)).argmax() for sample in majority_vote_array])

# Calculamos la precisión del voto mayoritario
print("Accuracy: ", accuracy_score(y, one_vs_one))

# Creamos una lista para almacenar los pares de clases
pairs = []
# Creamos un conjunto para almacenar las clases que aún no hemos visto
left_overs = set(np.unique(y))

# Iteramos a través de cada clase
for class_ in np.unique(y):
    # Eliminamos la clase que ya hemos visto
    left_overs.remove(class_)
    # La segunda clase en el par
    for second_class in left_overs:
        # Añadimos el par a la lista
        pairs.append(str(class_) + ' and ' + str(second_class))
        print("class {} vs class {} ".format(class_, second_class))
        # Creamos un array temporal para las etiquetas
        temp_y = np.zeros(y.shape)
        # Encontramos las clases en el par
        select = np.logical_or(y == class_, y == second_class)
        # Entrenamos el modelo
        model = SVC(kernel='linear', gamma=.5, probability=True)
        model.fit(X[select, :], y[select])
        # Añadimos el modelo a la lista
        my_models.append(model)
        # Dibujamos el límite de decisión para cada par y las muestras de entrenamiento correspondientes
        decision_boundary(X[select, :], y[select], model, iris, two=True)

print(pairs)
