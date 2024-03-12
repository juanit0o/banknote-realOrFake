import numpy as np
from sklearn.utils import shuffle
                                            #####  Preprocess data #####

def preProcessData(fileNameTrain, fileNameTest):
    #Train data processing
    matTrain = np.loadtxt(fileNameTrain, delimiter="\t")
    dataShuffledTrain = shuffle(matTrain)
    trainClasses = dataShuffledTrain[:,-1] #0 - real ou 1 - fake
    trainFeatures = dataShuffledTrain[:,0:-1] #todas as 4 features

    aux = []
    for a in trainClasses:
        aux.append([a])
    trainingClasses = np.array(aux)

    meansTrain = np.mean(trainFeatures,axis=0) #axis = 0 media ao longo das colunas, axis = 1 media ao longo das linhas
    stDevsTrain = np.std(trainFeatures, axis=0)
    trainFeatures = (trainFeatures - meansTrain)/stDevsTrain #temos agora as features standardizadas

    matTrainStandardized = np.append(trainFeatures, trainingClasses, axis = 1)

    #Test data processing
    matTest = np.loadtxt(fileNameTest, delimiter="\t")
    dataShuffledTest = shuffle(matTest)

    testClasses = dataShuffledTest[:,-1] #0 - real ou 1 - fake
    testFeatures = dataShuffledTest[:,0:-1] #todas as 4 features

    testFeatures = (testFeatures - meansTrain)/stDevsTrain #temos agora as features standardizadas

    aux = []
    for a in testClasses:
        aux.append([a])
    testingClasses = np.array(aux)


    matTestStandardized = np.append(testFeatures, testingClasses, axis = 1)
    return trainFeatures, trainClasses, testFeatures, testClasses, matTrainStandardized, matTestStandardized, dataShuffledTest, dataShuffledTrain