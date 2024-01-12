# Klasifikasi karakter tulisan tangan pada dataset MNIST dengan HOG Feature Extraction dan SVM

# Import library
import tensorflow as tf 
import numpy as np 
from tensorflow.keras import datasets # Library untuk membangun dan melatih model menggunakan TensorFlow
from skimage.feature import hog # Library untuk ekstraksi fitur HOG
from sklearn.svm import SVC # Library untuk klasifikasi dengan SVM
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix # Library untuk evaluasi performa model 

# Load data
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data() # Meload data dari dataset MNIST

# Ekstraksi Fitur HOG untuk data latih
hog_features_train = [hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3,3)) for image in x_train]
hog_features_train = np.array(hog_features_train) # Mengubah hog_features_train menjadi array numpy

# Ekstraksi fitur HOG untuk data test
hog_features_test = [hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3,3)) for image in x_test]
hog_features_test = np.array(hog_features_test) # Mengubah hog_features_test menjadi array numpy

# Membuat dan melatih model SVM
svm_model = SVC(gamma='scale') # Membuat model SVM
svm_model.fit(hog_features_train, y_train) # Melatih model SVM dengan data fitur dan label dari dataset latih

# Prediksi dan evaluasi
svm_predictions = svm_model.predict(hog_features_test) # Memprediksi dataset test menggunakan model SVM yang sudah dilatih

# Menampilkan hasil evaluasi
print("Confusion Matrix:", confusion_matrix(y_test, svm_predictions)) # Menampilkan hasil prediksi confusion matrix
print("Accuracy:", accuracy_score(y_test, svm_predictions)) # Menampilkan hasil prediksi accuracy
print("Precision:", precision_score(y_test, svm_predictions, average='weighted')) # Menampilkan hasil prediksi precision
