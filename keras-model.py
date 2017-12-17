import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model

X = np.load("data/data.npy")
y = to_categorical(np.load("data/targets.npy"))

X = np.array([row.flatten() for row in X])

model = Sequential()
model.add(Dense(500, input_dim=X.shape[1], activation='selu'))
model.add(Dense(2, input_dim=X.shape[1], activation='softmax'))
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

data = model.fit(X, y, epochs=10, validation_split=0.1).history

for layer in model.layers:
    im = layer.get_weights()[0]
    plt.imshow(X @ im, cmap=plt.cm.seismic)
    plt.title("Visualisation of Neural Network weights")
    plt.ylabel("Flattened hidden layer input node")
    plt.xlabel("Hidden layer output neuron")
    plt.show()

plt.xlabel('Hidden Layer Units')
plt.ylabel('Accuracy')

plt.legend()

plt.show()
