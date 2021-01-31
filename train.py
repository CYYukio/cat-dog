import h5py
import numpy as np
from sklearn.utils import shuffle
from keras.models import *
from keras.layers import *
import pandas as pd
from keras.preprocessing.image import *
import matplotlib.pylab as plt
np.random.seed(2017)

BATCH_SIZE=128
EPOCHS=40


X_train = []
X_test = []

for filename in ["gap_ResNet50.h5", "gap_VGG19.h5", "gap_InceptionV3.h5"]:
    with h5py.File(filename, 'r') as h:
        X_train.append(np.array(h['train']))
        X_test.append(np.array(h['test']))
        y_train = np.array(h['label'])

X_train = np.concatenate(X_train, axis=1)
X_test = np.concatenate(X_test, axis=1)

X_train, y_train = shuffle(X_train, y_train)


input_tensor = Input(X_train.shape[1:])
x = Dropout(0.5)(input_tensor)
x = Dense(1, activation='sigmoid')(x)
model = Model(input_tensor, x)

model.compile(optimizer='adadelta',
              loss='binary_crossentropy',
              metrics=['accuracy'])

_history=model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2)

plt.style.use("ggplot")
plt.figure()
N= EPOCHS
plt.plot(np.arange(0, N), _history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), _history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), _history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), _history.history["val_accuracy"], label="val_acc")
plt.title("loss and accuracy")
plt.xlabel("epoch")
plt.ylabel("loss/acc")
plt.legend(loc="best")
plt.savefig("./result.png")
plt.show()