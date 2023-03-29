import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Inyecion de archivos
Entrada = pd.read_excel('Entrada.xlsx')
Salida = pd.read_excel('Salida.xlsx')

scaler = StandardScaler()
x_entrada = Entrada
y_salida = Salida
x_entrada = scaler.fit_transform(x_entrada)

model = keras.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=(x_entrada.shape[1],)),
    keras.layers.Dense(60, activation='relu'),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae', keras.metrics.RootMeanSquaredError()])
regularization = keras.regularizers.l2(0.01)
model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=regularization))
model.add(keras.layers.Dense(1, activation='linear', kernel_regularizer=regularization))

#Entrenamiento de la red neuronal
history =model.fit(x_entrada, y_salida, epochs=50, batch_size=32)

prueba_x = Entrada
prueba_y = Salida
# Evaluar datos
scores = model.evaluate(prueba_x, prueba_y)
predictions = model.predict(prueba_x)

for x in predictions:
    if(int(round(x[0]))==1):
            print(int(round(x[0]))," Positivo Para Malaria")
    else:
          print(int(round(x[0]))," Negativo Para Malaria")


print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))