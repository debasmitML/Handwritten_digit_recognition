# Handwritten_digit_recognition
Handwritten digit classification is done by two methods
1. Using ML:
From scikit-learn mnist handwritten digit dataset has been taken and then t-SNE has been used to reduce the dimensions into 16 features. t-SNE has been used because it clusters different classes by measuring probabilistic distance metrics. Then Normalization and feature elimination has been done. 1000 images have been taken for training and 300 images have been taken for testing. Then 4 ML classifiers namely SVM., Random Forest, KNN and Logistic Regression have been deployed for the classification. SVM gives maximum accuracy, so SVM model has been used for testing using POSTMAN.

2. using CNN:
Tensorflow handwwritten digit dataset has been taken and and allthe images have been normalised. ResNet like architecture has been built for the classification