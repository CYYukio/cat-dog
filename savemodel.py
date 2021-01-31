##准备用预训练过的大型网络
from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *

import h5py

def save_model(MODEL,image_size,lambda_func=None):
    width = image_size[0]
    height = image_size[1]
    input_tensor = Input((width, height, 3))
    x = input_tensor

    if lambda_func:
        x=Lambda(lambda_func)(x)

    base_model=MODEL(input_tensor=x,weights='imagenet',include_top=False)
    model=Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

    gen=ImageDataGenerator()
    train_generator=gen.flow_from_directory("train2",image_size, shuffle=False, batch_size=16)
    test_generator = gen.flow_from_directory("test2", image_size, shuffle=False, batch_size=16,class_mode=None)

    train=model.predict(train_generator,train_generator.samples)
    test=model.predict(test_generator,test_generator.samples)

    with h5py.File("pre_%s.h5" % MODEL.func_name) as h:
        h.create_dataset("train", data=train)
        h.create_dataset("test", data=test)
        h.create_dataset("label", data=train_generator.classes)


save_model(ResNet50, (224, 224))
#save_model(Xception, (299, 299), xception.preprocess_input)
