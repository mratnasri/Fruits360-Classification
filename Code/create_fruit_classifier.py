import os
import os.path
from sklearn.datasets import load_files
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dense, Flatten
from keras.optimizers import SGD
from sklearn.model_selection import KFold
from numpy import mean, std

# 100x100 images
# 131 classes

#train_dir = '../../trial data/Training'
#test_dir = '../../trial data/Test'
train_dir = '../../fruits-360/Training'
test_dir = '../../fruits-360/Test'
train_categories = []
categories_n = 0
test_categories = []
train_n = []
train_total = 0
test_n = []
test_total = 0

for label in os.listdir(train_dir):
    train_categories.append(label)
    train_n.append(len(os.listdir(train_dir + "/" + label)))

train_total = sum(train_n)

for label in os.listdir(test_dir):
    test_categories.append(label)
    test_n.append(len(os.listdir(test_dir + "/" + label)))

test_total = sum(test_n)
print("Number of train categories = ", len(train_categories))
print("Number of train samples = ", train_total)
print("Number of test categories = ", len(test_categories))
print("Number of test samples = ", test_total)
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
x_test, y_test, _ = load_dataset(test_dir)

y_train = to_categorical(y_train, categories_n)
y_test = to_categorical(y_test, categories_n)

# preprocessing


def convert_img_to_array(dir):
    img_array = []
    for file in dir:
        img_array.append(img_to_array(load_img(file)))
    return img_array


x_train = np.array(convert_img_to_array(x_train))
print('Training set shape : ', x_train.shape)

x_test = np.array(convert_img_to_array(x_test))
print('Test set shape : ', x_test.shape)

# normalization
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255


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
                  metrics=['accuracy'])
    return model

# evaluate model using 5-fold cross validation


def evaluate_model(datax, datay, n_folds=5):
    scores, histories = list(), list()
    # prepare cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    # enumerate splits
    i = 1
    for train_ix, test_ix in kfold.split(datax):
        # define model
        model = model_config()
        # select rows for train and test
        trainx, trainy, valx, valy = datax[train_ix], datay[train_ix], datax[test_ix], datay[test_ix]
        # fit model
        history = model.fit(trainx, trainy, epochs=10, batch_size=32,
                            validation_data=(valx, valy), verbose=0)
        # evaluate model
        _, acc = model.evaluate(valx, valy, verbose=0)
        print(i, '. %.3f' % (acc * 100.0), sep='')
        i += 1
        # stores scores
        scores.append(acc)
        histories.append(history)
    print('Accuracy: mean=%.3f std=%.3f, n=%d' %
          (mean(scores)*100, std(scores)*100, len(scores)))
    return scores, histories, model


scores, histories, model = evaluate_model(x_train, y_train)
model.save('../Models/fruits_keras_model.h5')
print("saved")
