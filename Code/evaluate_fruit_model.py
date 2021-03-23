import os
import os.path
from sklearn.datasets import load_files
import numpy as np
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, top_k_accuracy_score, ConfusionMatrixDisplay
import sys
from matplotlib import pyplot as plt

#test_dir = '../../trial data/Test'
test_dir = '../../fruits-360/Test'
test_categories = []
test_n = []
test_total = 0
for label in os.listdir(test_dir):
    test_categories.append(label)
    test_n.append(len(os.listdir(test_dir + "/" + label)))

test_total = sum(test_n)
print("Number of test categories = ", len(test_categories))
print("Number of test samples = ", test_total)
categories_n = len(test_categories)


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


x_test, y_test, target_labels = load_dataset(test_dir)
y_test_ohe = to_categorical(y_test, categories_n)

# preprocessing


def convert_img_to_array(dir):
    img_array = []
    for file in dir:
        img_array.append(img_to_array(load_img(file)))
    return img_array


x_test = np.array(convert_img_to_array(x_test))
print('Test set shape : ', x_test.shape)

# normalization

x_test = x_test.astype('float32')/255

# load model
model = load_model('../Models/fruits_keras_model6.h5')
# evaluate model on test dataset
pred_prob = model.predict(x_test)
#pred_class = model.predict_classes(x_test)
pred_class = np.argmax(pred_prob, axis=-1)
# reduce to 1D array
# pred_class = pred_class[:, 0]
# print(pred_class)
# metrics
accuracy = accuracy_score(y_test, pred_class)
k_accuracy = top_k_accuracy_score(y_test, pred_prob, k=5)
print('accuracy =  %.3f' % (accuracy * 100.0),
      'top-5 accuracy = %.3f' % (k_accuracy*100))
"""precision = precision_score(y_test, pred_class)
recall = recall_score(y_test, pred_class)"""
report = classification_report(y_test, pred_class)
print("Classification Report: ")
print(report)
"""f1 = f1_score(y_test, pred_class,average='macro')
print("f1 score: ", f1)"""
confusionMatrix = confusion_matrix(
    y_test, pred_class)  # row(true), column(predicted)
np.set_printoptions(threshold=sys.maxsize)
print("Confusion matrix: ")
print(confusionMatrix)
np.set_printoptions(threshold=False)
disp = ConfusionMatrixDisplay(
    confusion_matrix=confusionMatrix, display_labels=target_labels)
disp.plot()
plt.show()

"""# loss,accuracy, other metrics...
_, acc, k_acc = model.evaluate(x_test, y_test_ohe, verbose=0)
print('accuracy =  %.3f' % (acc * 100.0),
      'top-5 accuracy = %.3f' % (k_acc*100))"""
