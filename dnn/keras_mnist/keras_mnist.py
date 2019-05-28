# -*- coding: utf-8 -*-
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import l2



# load data and configure parameters

x_dim = 28*28
num_classes = 10
batch_size = 50
epochs=1

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# reshape and normalize x
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
# convert y to one-hot form
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)



# define model
model = Sequential()
model.add(Dense(units=256, input_dim=x_dim, use_bias=True, activation='relu', name='dense1'))
#model.add(Dropout(rate=0.5))
model.add(Dense(units=128, input_dim=256, use_bias=True, activation='tanh'))
#model.add(Dropout(rate=0.5))
model.add(Dense(units=num_classes, input_dim=128, use_bias=True, activation='softmax'))

# plot model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
img = mpimg.imread('model.png')
plt.imshow(img)
plt.axis('off')
plt.show()

# print model
model.summary()



# train model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
loss, acc = model.evaluate(x_test, y_test, verbose=1)
print('Loss:', loss, 'Acc:', acc)





