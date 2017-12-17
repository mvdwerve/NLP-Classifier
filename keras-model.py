import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.utils import to_categorical
import matplotlib.pyplot as plt

X = np.load("data/data.npy")
y = to_categorical(np.load("data/targets.npy"))

X = np.array([row.flatten() for row in X])
results =[]
vresults =[]
reserr = []
vreserr = []

a = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 3000]
for i in a:
    avg = []
    vavg = []
    for j in range(4):
        model = Sequential()
        model.add(Dense(i, input_dim=X.shape[1], activation='selu'))
        model.add(Dense(2, input_dim=X.shape[1], activation='softmax'))
        model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        data = model.fit(X, y, epochs=10, validation_split=0.1).history
        acc, valacc = data['acc'][-1], data['val_acc'][-1]

        avg.append(acc) 
        vavg.append(valacc)

    results.append(np.mean(avg))
    vresults.append(np.mean(vavg))

    reserr = np.std(avg)
    vreserr = np.std(vavg)

plt.errorbar(a, results, reserr, label='Training set')
plt.errorbar(a, vresults, vreserr, label='Testing set')

plt.xlabel('Hidden Layer Units')
plt.ylabel('Accuracy')

plt.legend()

plt.show()
