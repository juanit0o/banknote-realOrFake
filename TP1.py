#NB.png - The plot of training and cross-validation errors for the KDE kernel width of your implementation of the Naïve Bayes classifier.
#SVM.png - The plot of training and cross-validation errors for the gamma parameter of the SVM classifier.

#Objective
#The goal of this assignment is to parametrize, fit and compare Naive Bayes and Support Vector Machine classifiers. 

#Attribute Information:
#1. variance of Wavelet Transformed image (continuous)
#2. skewness of Wavelet Transformed image (continuous)
#3. curtosis of Wavelet Transformed image (continuous)
#4. entropy of image (continuous)
#5. class (integer)

#You must implement your own Naïve Bayes classifier using Kernel Density Estimation for the probability distributions of the feature values. 
#For this, you can use any code from the lectures, lecture notes and tutorials that you find useful. 
# Also, use the KernelDensity class from sklearn.neighbors.kde for the density estimation.
#You will need to find the optimum value for the bandwitdh parameter of the kernel density estimators you will use.
#  Use the training set provided in the TP1_train.tsv for this.

# The second classifier will be the Gaussian Naïve Bayes classifier in the sklearn.naive_bayes.GaussianNB class. 
# You DO NOT need to adjust aditional parameters for this classifier

#Finally, use a Support Vector Machine with a Gaussian radial basis function, available in the sklearn.svm.SVC class. 
# Use a regularization factor C = 1 and OPTIMIZE THE GAMMA PARAMETER with CROSS VALIDATION on the training set.

#After training your classifiers and fine tuning all parameters, COMPARE PERFORMANCE OF THE THREE CLASSIFIERS,
# identify the best one and discuss if it is significantly better than the others.

#The data are available on .tsv files where each line corresponds to a bank note and the five values, separated by commas, are, in order,
# the four features (variance, skewness and curtosis of Wavelet Transformed image and the entropy of the bank note image) and the class label,
#  an integer with values 0 for real bank notes and 1 for fake bank notes.

#In addition to the code, you must include TWO PLOTS with the TRAINING and CROSSVALIDATION errors for the OPTIMIZATION OF THE BANDWITH PARAMETER
# of your Naïve Bayes classifier (this plot should be named NB.png) and the gamma value of the SVM classifier 
# (this plot should be named SVM.png). These plots should have a legend identifying which line corresponds to which error.

#TIPS
#Process the data correctly, RANDOMIZE ORDER OF DATA POINTS and STANDARDIZE VALUES.
#Determine the PARAMETERS WITH CROSS VALIDATION ON THE TRAINING SET, leaving the TEST SET FOR FINAL COMPARISONS.
#USE THE SAME BANDWITH VALUE FOR ALL THE KDE (Kernel Density Estimators) in each instance of your Naïve Bayes classifier, 
# and try values from 0.02 to 0.6 with a step of 0.02 for the bandwidth.

#Use the DEFAULT KERNEL ('gaussian') for the Kernel Density Estimators.
#To optimize the gamma parameter of the SVM, try values from 0.2 to 6 with a step of 0.2
#When splitting your data for cross validation use stratified sampling.
#Use 5 folds for cross validation
#Use the fraction of incorrect classifications as the measure of the error. This is equal to 1-accuracy, and the accuracy can be obtained with the score method of the classifiers provided by the Scikit-learn library.
#For your Naïve Bayes classifier, you can implement your own measure of the accuracy or use the accuracy_score function in the metrics module if you find it useful.
#For comparing the classifiers, use the approximate normal test and McNemar's test, both with a 95% confidence interval


import numpy as np
import math
from itertools import combinations
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import matplotlib.pyplot as plot
from sklearn.neighbors import KernelDensity
import NaiveBayes
import preProcessData



                                            #####  Preprocess data #####

trainFeatures, trainClasses, testFeatures, testClasses,\
     matTrainStandardized, matTestStandardized, dataShuffledTest, dataShuffledTrain = preProcessData.preProcessData("tp1_train.tsv", "tp1_test.tsv")

test_size = len(testClasses)
                                            #############################
bandwiths = np.arange(0.02,0.6,0.02) #0.02

folds = 5
stratKFold = StratifiedKFold(n_splits = folds)
trainingErrors = []
validationErrors = []

classified = dataShuffledTrain[:,[4]].T[0]

for bandwith in bandwiths:
    # Classes of training set
    errorTrain = 0
    errorValidation = 0
    
    for trainIndex,validationIndex in stratKFold.split(classified, classified):     
        trainSet = matTrainStandardized[trainIndex]
        validationSet = matTrainStandardized[validationIndex]
        
        kde = NaiveBayes.fit(trainSet, bandwith)
        
        errorTrain += NaiveBayes.score_fold(trainSet, trainSet, kde)[0]
        errorValidation += NaiveBayes.score_fold(trainSet, validationSet, kde)[0]
    
    trainingErrors.append(errorTrain / folds)
    validationErrors.append(errorValidation / folds)
    
minError = np.min(validationErrors)
bestBoyBandwith = bandwiths[validationErrors.index(minError)]
bestKde = NaiveBayes.fit(matTrainStandardized, bestBoyBandwith)
print("Best Validation Error: ", minError)
print("Best Bandwith: ", bestBoyBandwith)

#Erro do test com o melhor gamma escolhido no cross validation
errorTestNB, num_of_errorsNB, success_colNB = NaiveBayes.score_fold(matTrainStandardized, matTestStandardized, bestKde)
print("Test Error: ", errorTestNB)
print("Number of Errors (NB):", num_of_errorsNB)

plotTrain, = plot.plot(bandwiths, trainingErrors, 'b', label='Training error')
plotVal, = plot.plot(bandwiths, validationErrors, 'g', label='Validation error')
plots = [plotTrain, plotVal]

plot.text(10, 10, 'matplotlib')
plot.legend(plots, ['Training Error', 'Validation Error'], loc='upper center', fancybox=True, shadow=True)
plot.xlabel("Bandwith")
plot.annotate("Test Error: " + str(round(errorTestNB, 4)) + "\nBest Bandwith: " + str(round(bestBoyBandwith, 3)), xy=(280, 30), xycoords='axes points', size=10, ha='left', va='top', bbox=dict(boxstyle='round', fc='w'))
plot.savefig('NB.png', dpi=300) 
plot.close()

                                            #############################

# The second classifier will be the Gaussian Naïve Bayes classifier in the sklearn.naive_bayes.GaussianNB class. 
# You DO NOT need to adjust aditional parameters for this classifier

gaussianNBClassifier = GaussianNB()
#Treino
gaussianNBClassifier.fit(trainFeatures, trainClasses)
trainErrorGaussianNB = 1-gaussianNBClassifier.score(trainFeatures, trainClasses)
#Teste
predictGaussianNB = gaussianNBClassifier.predict(testFeatures)
testErrorGaussianNB = 1-gaussianNBClassifier.score(testFeatures, testClasses)
######## Resultados finais GNB #####
num_of_errorsGNB = round(testErrorGaussianNB * len(testClasses), 0)
print("Gaussian Naive Bayes - Training Error:\t" + str(round(trainErrorGaussianNB,3))) 
print("Gaussian Naive Bayes - Test Error:\t" + str(round(testErrorGaussianNB,3)))

                                            #############################

#Finally, use a Support Vector Machine with a Gaussian radial basis function, available in the sklearn.svm.SVC class. 
# Use a regularization factor C = 1 and OPTIMIZE THE GAMMA PARAMETER with CROSS VALIDATION on the training set.

gammaSequence = np.arange(0.2,6,0.2)
allTrainErrorsSVM = []
allValidationErrorsSVM = []
bestError = math.inf
bestGamma = 0

for gamma in gammaSequence:
    trainErrorSVM = 0
    validationErrorSVM = 0
    for trainIndex,validationIndex in stratKFold.split(trainClasses,trainClasses):
        trainFeaturesSet = trainFeatures[trainIndex]
        trainClassesSet = trainClasses[trainIndex]
        validationFeaturesSet = trainFeatures[validationIndex]
        validationClassesSet = trainClasses[validationIndex]

        #kernel gaussian radial basis function 
        sv = svm.SVC(C=1, kernel = "rbf", gamma=gamma)
        sv.fit(trainFeaturesSet,trainClassesSet)

        #erro treino
        trainErrorSVM += (1-sv.score(trainFeaturesSet, trainClassesSet))
        #erro validacao
        validationErrorSVM += (1-sv.score(validationFeaturesSet, validationClassesSet))
    #erros medios para o treino e validacao (media p todos os folds)
    trainErrorSVM = trainErrorSVM/folds
    validationErrorSVM = validationErrorSVM/folds

    if validationErrorSVM < bestError:
        bestGamma = gamma
        bestError = validationErrorSVM

    #erros no vetor para o plot pedido
    allTrainErrorsSVM.append(trainErrorSVM)
    allValidationErrorsSVM.append(validationErrorSVM)

print("Best Gamma SVM:" ,bestGamma, "\nBest Validation Error", bestError)

sv = svm.SVC(C=1, kernel = "rbf", gamma=bestGamma)
sv.fit(trainFeatures, trainClasses)
predictSVM = sv.predict(testFeatures)
testErrorSVM = (1-sv.score(testFeatures, testClasses))
num_of_errorsSVM = round(testErrorSVM * len(testClasses), 0)

print("Number of Errors (SVM):", num_of_errorsSVM)
print("Test error SVM: " + str(testErrorSVM))


plot.plot(gammaSequence, allTrainErrorsSVM, 'b', label='Training error')
plot.plot(gammaSequence, allValidationErrorsSVM, 'g', label='Validation error')
plot.legend()
plot.savefig('SVM.png', dpi=300)
plot.close()


############ MCNemar
def approxNormalTest(X, N):
    return math.sqrt(X * (1 - (X/N)))

def approximateNormalTest (X_NB, X_GNB=1, X_SVM=1, z=1.96,  N_NB=test_size, N_GNB=test_size, N_SVM=test_size):
    sigmaErrorNB = approxNormalTest(X_NB,N_NB)
    sigmaErrorGNB = approxNormalTest(X_GNB, N_GNB)
    sigmaErrorSVM = approxNormalTest(X_SVM, N_SVM)
    confidenceNB = (X_NB, z * sigmaErrorNB) 
    confidenceGNB = (X_GNB, z * sigmaErrorGNB)
    confidenceSVM = (X_SVM, z * sigmaErrorSVM)
    return confidenceNB, confidenceGNB, confidenceSVM

def computeInterval(num_errors, confidence):
    return num_errors - confidence,  num_errors + confidence

def overlaps(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0])) 
        

def compareClassifiers(list_classifiers):
    list_intervals = []

    for c in list_classifiers:
        list_intervals.append( (computeInterval(c[0][0], c[0][1]),  c[1]) )

    for a in list(combinations(list_intervals, 2)):
        if overlaps(a[0][0], a[1][0]):
            print(a[0][1] + " and " + a[1][1] + " overlap!")
        else:
            if a[0][0][0] < a[1][0][0]:
                print(a[0][1] + " is better than " + a[1][1] + " with 95% confidence!")
            else:
                print(a[1][1] + " is better than " + a[0][1] + " with 95% confidence!")


confidenceNB, confidenceGNB, confidenceSVM = approximateNormalTest(num_of_errorsNB, num_of_errorsGNB, num_of_errorsSVM)
print("\nNaive Bayes: ", confidenceNB[0], "+-", confidenceNB[1])
print("Gaussian Naive Bayes: ", confidenceGNB[0], "+-", confidenceGNB[1])
print("SVM: ", confidenceSVM[0], "+-", confidenceSVM[1], "\n")

compareClassifiers([(confidenceNB, "Naive Bayes"), (confidenceGNB, "Gaussian Naive Bayes"), (confidenceSVM, "Support Vector Machine")])


def getE01(col_1, col_2):
    e01 = 0
    for i in range(0, min(len(col_1), len(col_2))):
        if col_1[i] == 0 and col_2[i] == 1:
            e01 += 1
    return e01

def mcNemarTest(classifier_1, classifier_2):
    e01 = getE01(classifier_1, classifier_2)
    e10 = getE01(classifier_2, classifier_1)

    return ((abs(e01 - e10) - 1)**2)/(e01+e10)

#Compare predict vector with real classes
success_colGNB = predictGaussianNB == testClasses
success_colSVM = predictSVM == testClasses


mcNemar_NBGNB = round(mcNemarTest(success_colNB, success_colGNB), 4)
mcNemar_NBSVM= round(mcNemarTest(success_colNB, success_colSVM), 4)
mcNemar_GNBSVM = round(mcNemarTest(success_colGNB, success_colSVM), 4)

print("\nMcNemar Tests:\nNaive Bayes vs Gaussian Naive Bayes = ", mcNemar_NBGNB, "\nNaive Bayes vs SVM = ", mcNemar_NBSVM, "\nGaussian Naive Bayes vs SVM = ", mcNemar_GNBSVM)

######### Optional, melhor combinacao de gamma e C

gammaSequence = np.arange(0.2, 6,0.2)
Csequence = np.arange(1, 200, 10)
allTrainErrorsSVM = []
allValidationErrorsSVM = []
bestError = math.inf
bestGamma = 0
bestCee = 0

for gamma in gammaSequence:
    for cee in Csequence:
        trainErrorSVM = 0
        validationErrorSVM = 0

        for trainIndex,validationIndex in stratKFold.split(trainClasses,trainClasses):
            trainFeaturesSet = trainFeatures[trainIndex]
            trainClassesSet = trainClasses[trainIndex]
            validationFeaturesSet = trainFeatures[validationIndex]
            validationClassesSet = trainClasses[validationIndex]

            #kernel gaussian radial basis function 
            sv = svm.SVC(C=cee, kernel = "rbf", gamma=gamma)
            sv.fit(trainFeaturesSet,trainClassesSet)

            #erro treino
            trainErrorSVM += (1-sv.score(trainFeaturesSet, trainClassesSet))
            #erro validacao
            validationErrorSVM += (1-sv.score(validationFeaturesSet, validationClassesSet))
        
        #erros medios para o treino e validacao (media p todos os folds)
        trainErrorSVM = trainErrorSVM/folds
        validationErrorSVM = validationErrorSVM/folds
        
        #erros no vetor para o plot pedido
        allTrainErrorsSVM.append(trainErrorSVM)
        allValidationErrorsSVM.append(validationErrorSVM)

        if validationErrorSVM < bestError:
            bestGamma = gamma
            bestCee = cee
            bestError = validationErrorSVM

print("Best Gamma SVM: " ,bestGamma, "\nBest Validation Error: ", bestError, "\nBest C: ", bestCee)

sv = svm.SVC(C=bestCee, kernel = "rbf", gamma=bestGamma)
sv.fit(trainFeatures, trainClasses)
predictSVMNovo = sv.predict(testFeatures)
testErrorSVM = (1-sv.score(testFeatures, testClasses))
num_of_errorsSVMO = round(testErrorSVM * len(testClasses), 0)
print("Num of errors SVMO:", num_of_errorsSVMO)

print("\nTest Error with best gamma & C: ", testErrorSVM)

confidenceSVMO, a,b = approximateNormalTest(num_of_errorsSVMO)
print("\nSupport Vector Machine Otimizado: ", confidenceSVMO[0], "+-", confidenceSVMO[1])
compareClassifiers([(confidenceSVMO, "Support Vector Machine Otimizado"), (confidenceSVM, "Support Vector Machine")])
