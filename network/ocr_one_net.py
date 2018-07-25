import numpy as np
from PIL import Image, ImageFont, ImageDraw
from keras.models import Sequential
from keras import optimizers, losses, layers, models
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.utils import to_categorical
from configuration.config import PATH
import matplotlib.pyplot as plt
import os
import keras


def create_model(cls):
    k_size = (5, 5)
    p_size = (2, 2)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=k_size,
                        border_mode='valid',
                        input_shape=(48, 48, 1),
                        activation='relu'))
    model.add(MaxPool2D(pool_size=p_size))
    model.add(Dropout(0.3))

    model.add(Conv2D(32, kernel_size=k_size, activation='relu'))
    model.add(MaxPool2D(pool_size=p_size))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.6))

    model.add(Dense(cls+1))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=0.001),
                  metrics=['accuracy'])
    return  model

def load_data(data_path, cls, islast=False):
    print('Loading data...')
    x = np.load(data_path).reshape(-1,48,48,1)
    batch = len(x)/48*48
    if islast:
        y = np.zeros([int(batch), cls+1])
        y[:, cls] = 1
        print(y)
    else:
        y = to_categorical(list(range(cls)) * int(batch/cls), cls+1)
    return x, y

from keras import callbacks
def train(data_path, model_name, cls=3557, batch_size=1024, epochs=30):
    if not os.path.exists(PATH.MODEL_DIR + model_name):
        print('Creating new model...')
        model = create_model(cls)
    else:
        print('Loading model...')
        print(PATH.MODEL_DIR + model_name)
        model = keras.models.load_model(PATH.MODEL_DIR + model_name)

    x_, y_ = load_data(data_path[1], cls, islast=True)
    x, y = load_data(data_path[0], cls)
    x = np.concatenate([x, x_], axis=0)
    y = np.concatenate([y, y_], axis=0)
    weight = ((cls - np.arange(cls)) / (cls) + 1) ** 2
    weight = list(weight)
    weight.append(4)
    weight = np.array(weight)
    weight = dict(zip(range(cls+1), weight / weight.mean()))
    log = callbacks.CSVLogger(PATH.LOG_DIR + 'log.csv')
    checkpoint = callbacks.ModelCheckpoint(PATH.MODEL_DIR + model_name, monitor='acc',
                                           save_best_only=True, save_weights_only=False, verbose=1)
    #lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: 0.001 * (0.9 ** epoch))
    test_data = PATH.DATASET_DIR + 'val_data.npy'
    tx, ty = load_data(test_data, cls)
    history = model.fit(x, y,
                        batch_size=batch_size,
                        epochs=epochs,
                        class_weight=weight,
                        callbacks=[log, checkpoint],
                        validation_data=(tx, ty))

    return history

def eval(model_name, test_data):
    print('Loading model...')
    model = keras.models.load_model(PATH.MODEL_DIR+model_name)
    tx,ty = load_data(test_data, 3557)
    loss, acc = model.evaluate(tx, ty)
    print('Loss: %s, Acc: %s' % (loss, acc))

def plot(history):
    acc, loss = history['acc'], history['loss']
    p1, = plt.plot(acc)
    p2, = plt.plot(loss)
    plt.ylabel('Accuracy && Loss')
    plt.xlabel('Epoch')
    plt.title('Optimization')
    plt.legend([p1,p2],['Accuracy','Loss'],loc='upper left')

    #plt.text(0, acc[0], 'Accuracy')
    #plt.text(0, loss[0], 'Loss')
    plt.savefig(PATH.LOG_DIR+'result.png')
    plt.show()

def main():
    train_data1 = PATH.DATASET_DIR+'test7x1.npy'
    train_data2 = PATH.DATASET_DIR+'3558cls.npy'
    model_name = 'model_keras/model.h5'
    history = train(data_path=[train_data1,train_data2], model_name=model_name, epochs=50)
    #plot(history.history)

    test_data = PATH.DATASET_DIR + 'test7x1.npy'
    test_data = PATH.DATASET_DIR + 'val_data.npy'
    eval(model_name, test_data)


if __name__ == '__main__':
    main()