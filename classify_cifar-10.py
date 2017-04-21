from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.utils.visualize_util import plot, model_to_dot
nb_epoch = 200
batch_size = 32
nb_classes = 10
img_rows, img_cols = 32, 32


(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.reshape(50000, 3072)#33*33*3 = 3072
X_test = X_test.reshape(10000, 3072)
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(4096, input_shape=(3072, )))
model.add(Activation('softmax'))
model.add(Dense(2048))
model.add(Activation('tanh'))
model.add(Dropout(0.35))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
#dot = model_to_dot(model).create(prog='dot')
plot(model, show_shapes=True, to_file='model.png')
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# model.fit(X_train, Y_train, batch_size=batch_size,
#           nb_epoch=nb_epoch, validation_data=(X_test, Y_test),
#           shuffle=True)

score = model.evaluate(X_test, Y_test, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])
print("Fin")
