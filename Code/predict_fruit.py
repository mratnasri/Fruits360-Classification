from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from sklearn.datasets import load_files
import numpy as np

# load and prepare the image

# test_dir = '../../trial data/Test'
test_dir = '../../fruits-360/Test'


def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(100, 100))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 1 channel
    img = img.reshape(1, 100, 100, 3)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img

# load an image and predict the class


def run_example():
    # load the image
    img = load_image('../../sample_fruit.jpg')
    # load model
    model = load_model('../Models/fruits_keras_model2.h5')
    # load label_names
    data_loading = load_files(test_dir)
    target_labels = np.array(data_loading['target_names'])
    # print("target labels size: ", len(target_labels)) #131
    # predict the class
    fruit = model.predict_classes(img)
    # print(fruit)
    print('predicted fruit: {0}, {1} '.format(
        fruit[0], target_labels[fruit[0]]))


# entry point, run the example
run_example()
