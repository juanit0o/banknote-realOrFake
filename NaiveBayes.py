# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 14:22:10 2021
@authors: André Costa, Joao Funenga
"""

import numpy as np
import math
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import matplotlib.pyplot as plot
from sklearn.neighbors import KernelDensity


                                            #####  Preprocess data #####


def fit(trainingData, h):
    #encontrar o melhor bandwith
    kdes = dict()
    kdes['real'] = dict()
    kdes['fake'] = dict()
    feature_number = 0
    
    classified = trainingData[:,4].T
    
    for feature in trainingData[:,[0,1,2,3]].T:
        feature_w_class_Fake = []
        feature_w_class_Real = []
        # A certain feature concatenated with the corresponding class
        for feature_num in range(0,len(classified)):
            if classified[feature_num] == 0:
                # All features of class "real" (where class=0)
                feature_w_class_Real.append([feature[feature_num]])
            else:
                # All features of class "fake" (where class=1)
                feature_w_class_Fake.append([feature[feature_num]])
        
        feature_w_class_Fake = np.array(feature_w_class_Fake)
        feature_w_class_Real = np.array(feature_w_class_Real)
        
        #KDE - Real Class
        kde_Real = KernelDensity(kernel='gaussian', bandwidth=h)
        kde_Real.fit(feature_w_class_Real)
        #KDE - Fake Class
        kde_Fake = KernelDensity(kernel='gaussian', bandwidth=h)
        kde_Fake.fit(feature_w_class_Fake)
        
        # Store log density of each kde
        kdes['real'][str(feature_number)] = kde_Real
        kdes['fake'][str(feature_number)] = kde_Fake
        feature_number += 1
        
    return kdes


def score_fold(training, validation, kdes):
    success_column = []
    
    classified = training[:,[4]].T[0]
    # Initialize these variables with the log of the prior
    log_real = np.log(len(classified[classified==0]) / len(classified))
    log_fake = np.log(len(classified[classified==1]) / len(classified))
    
    num_of_errors = 0
    
    for row in validation:
        real_sum = log_real
        fake_sum = log_fake
        for feature_num in range(0, len(row) - 1):
            sample = np.array([[row[feature_num]]])
            real_sum += kdes['real'][str(feature_num)].score_samples(sample)
            fake_sum += kdes['fake'][str(feature_num)].score_samples(sample)
        
        #real is more probable (class 0)           
        if real_sum > fake_sum:
            if row[-1] != 0:
                num_of_errors += 1
                success_column.append(0) # 0 if failed to predict
            else:
                success_column.append(1) # 1 if successful prediction
        #fake is more probable (class 1)
        else:
            if row[-1] != 1:
                num_of_errors += 1
                success_column.append(0) # 0 if failed to predict
            else:
                success_column.append(1) # 1 if successful prediction

    return num_of_errors / len(validation), num_of_errors, success_column
    