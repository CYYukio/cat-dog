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

for filename in ["pre_ResNet50.h5", "pre_VGG19.h5", "pre_InceptionV3.h5"]:
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
model.save("./model.h5")#保存模型

y_pred = model.predict(X_test, verbose=1)
y_pred = y_pred.clip(min=0.005, max=0.995)

df = pd.read_csv("sample_submission.csv",header=None, delim_whitespace=True, engine='python')

gen = ImageDataGenerator()
test_generator = gen.flow_from_directory("test2", (224, 224), shuffle=False,
                                         batch_size=16, class_mode=None)

for i, fname in enumerate(test_generator.filenames):
    index = int(fname[fname.rfind('/')+1:fname.rfind('.')])
    df.set_value(index-1, 'label', y_pred[i])

df.to_csv('pred.csv', index=None)
df.head(10)


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


