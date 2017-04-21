from keras.models import Sequential
from keras.layers import Dense
import numpy as np

'''A basic XOR simulation using a 3-2-out scheme'''
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], "float32")
y = np.array([[0], [1], [1], [0]], "float32")
w1 = np.array([[1, 1], [1, 1], ], "float32")  # !! Keras needs a matrix as weights everytime
w2 = np.array([[1], [-2]], "float32")
wb = np.array([0, -1], "float32")

# w = (w1,w2)
l = (w1, wb)
l2 = (w2, np.array([0]))
model = Sequential()
model.add(Dense(2, input_shape=(2,), activation='relu', weights=l))
model.add(Dense(1, activation='relu', weights=l2))

"""""
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])"""""

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# model.fit(x, y, nb_epoch=500, verbose=2)

print(model.get_weights())
print(model.predict(x).round())
