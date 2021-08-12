from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import DenseNet121
def model_arch():
    model = Sequential()
    base = DenseNet121(include_top = False, input_shape = (128, 128, 3), pooling = 'avg', classes = 2)
    for l in base.layers:
        l.trainable = False

    model.add(base)

    model.add(Flatten())
    model.add(BatchNormalization())

    model.add((Dense(256, activation = 'relu')))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add((Dense(128, activation = 'relu')))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add((Dense(64, activation = 'relu')))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add((Dense(32, activation = 'relu')))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add((Dense(16, activation = 'relu')))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add((Dense(1, activation = 'sigmoid')))

    opt = Adam()

    model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return  model
