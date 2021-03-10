import os
import os.path
from sklearn.datasets import load_files
import numpy as np
import keras
from keras.utils import to_categorical
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dense, Flatten
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from numpy import mean, std

# 100x100 images
# 131 classes

#train_dir = '../../trial data/Training'
train_dir = '../../fruits-360/Training'
train_categories = []
categories_n = 0
train_n = []
train_total = 0

for label in os.listdir(train_dir):
    train_categories.append(label)
    train_n.append(len(os.listdir(train_dir + "/" + label)))

train_total = sum(train_n)

print("Number of train categories = ", len(train_categories))
print("Number of train samples = ", train_total)

categories_n = len(train_categories)


def load_dataset(path):
    data_loading = load_files(path)
    #print("data loading: ", data_loading)
    filenames = np.array(data_loading['filenames'])
    #print("files add: ", files_add)
    targets = np.array(data_loading['target'])
    #print("target fruits: ", targets_fruits)
    target_labels = np.array(data_loading['target_names'])
    #print("target labels: ", target_labels_fruits)
    return filenames, targets, target_labels


x_train, y_train, target_labels = load_dataset(train_dir)
y_train = to_categorical(y_train, categories_n)

# split into training and validation
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.16, shuffle=True)
# preprocessing


def convert_img_to_array(dir):
    img_array = []
    for file in dir:
        img_array.append(img_to_array(load_img(file)))
    return img_array


x_train = np.array(convert_img_to_array(x_train))
print('Training set shape : ', x_train.shape)
x_val = np.array(convert_img_to_array(x_val))
print('Validation set shape : ', x_val.shape)

# normalization
x_train = x_train.astype('float32')
x_train = x_train/255
x_val = x_val.astype('float32')
x_val = x_val/255


def model_config():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_uniform', input_shape=(100, 100, 3)))

    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_uniform'))

    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(200, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(131, activation='softmax'))

    # compile the model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy()])
    return model

# evaluate model using 5-fold cross validation


def train_model(datax, datay, valx, valy):
    model = model_config()
    # fit model
    history = model.fit(datax, datay, epochs=10, batch_size=32,
                        validation_data=(valx, valy), verbose=2)
    print('Accuracy: mean=%.3f std=%.3f, n=%d' %
          (mean(history.history['accuracy'])*100, std(history.history['accuracy'])*100, len(history.history['accuracy'])))
    print('Top-5 Accuracy: mean=%.3f std=%.3f, n=%d' %
          (mean(history.history['top_k_categorical_accuracy'])*100, std(history.history['top_k_categorical_accuracy'])*100, len(history.history['top_k_categorical_accuracy'])))
    print('Validation Accuracy: mean=%.3f std=%.3f, n=%d' %
          (mean(history.history['val_accuracy'])*100, std(history.history['val_accuracy'])*100, len(history.history['val_accuracy'])))
    print('Validation Top-5 Accuracy: mean=%.3f std=%.3f, n=%d' %
          (mean(history.history['val_top_k_categorical_accuracy'])*100, std(history.history['val_top_k_categorical_accuracy'])*100, len(history.history['val_top_k_categorical_accuracy'])))

    return history, model


history, model = train_model(x_train, y_train, x_val, y_val)
model.save('../Models/fruits_keras_model3.h5')
print("saved")
