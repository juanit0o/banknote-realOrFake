# Classifier Comparison for Banknote Authentication

## Description
This project focuses on parametrizing, fitting, and comparing Naive Bayes and Support Vector Machine (SVM) classifiers for banknote authentication. The dataset used is inspired by the banknote authentication problem from the UCI Machine Learning Repository, with adaptations made specifically for this assignment.

## Data
- **Training Data:** [TP1_train.tsv](https://github.com/juanit0o/banknote-realOrFake/blob/main/TP1_train.tsv)
- **Test Data:** [TP1_test.tsv](https://github.com/juanit0o/banknote-realOrFake/blob/main/TP1_test.tsv)

## Instructions
1. Implement a Naïve Bayes classifier using Kernel Density Estimation for probability distributions of feature values. Utilize the `KernelDensity` class from `sklearn.neighbors.kde` for density estimation.
2. Find the optimum bandwidth parameter for the kernel density estimators using the training set.
3. Implement a Gaussian Naïve Bayes classifier using `sklearn.naive_bayes.GaussianNB`.
4. Use a Support Vector Machine with a Gaussian radial basis function, available in `sklearn.svm.SVC`. Use a regularization factor `C = 1` and optimize the `gamma` parameter with cross-validation on the training set.
5. Compare the performance of the three classifiers and identify the best one. Discuss if it significantly outperforms the others.
6. Fine-tune the SVM classifier by adjusting both `gamma` and `C` parameters simultaneously. This is worth 1/20 of the assignment grade.
   
## Data Format
- Each line in the `.tsv` files corresponds to a banknote.
- The values are comma-separated in the order: variance, skewness, curtosis, entropy, and class label (0 for real banknotes, 1 for fake banknotes).

## Additional Files
- **NB.png:** Plot of training and cross-validation errors for Naïve Bayes classifier optimization.
- **SVM.png:** Plot of training and cross-validation errors for SVM classifier optimization.
- **Questions:** Answer a set of questions about the assignment (choose either English or Portuguese version from the provided files).

## Contributors
- João Funenga & André Costa
