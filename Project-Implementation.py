# -*- coding: utf-8 -*-
"""
Created on Wed April 05 03:11:35 2018
@author: Nazmul C00409603
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import PolynomialFeatures
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

def main(): 
    #data = pd.read_csv('Final-train-dataset-all-without-Unig.csv', sep=',')
    data = pd.read_csv('Final-train-dataset-all-short.csv', sep=',')
    X = data.iloc[:,1:]
    Y = data.label
    Y1 = data.iloc[:,0]
    #print(Y) 
    print("Validation")
    #dataDev = pd.read_csv('Final-dev-dataset-all-without-unig.csv', sep=',')
    dataDev = pd.read_csv('Final-dev-dataset-all-short.csv', sep=',')
    XDev = dataDev.iloc[:,1:]
    YDev = dataDev.iloc[:,0]
    
    print("Testing")
    #dataTest = pd.read_csv('Final-test-dataset-all-without-unig.csv', sep=',')
    dataTest = pd.read_csv('Final-test-dataset-all-short.csv', sep=',')
    XTest = dataTest.iloc[:,1:]
    YTest = dataTest.iloc[:,0]
    #print(YTest)
    
    
    c = []
    
    model = GaussianNB()
    model.fit(X,Y)
    c=model.predict(XDev)
    accuracy = accuracy_score(YDev,c)
    print("Accuracy for Naive Bayes:", accuracy * 100, " %")
    #cm = confusion_matrix(YTest, c)
    #print("Confusion Matrix for Naive Bayes of Test data\n",cm)
    
    model4 = svm.SVC(decision_function_shape='ovo')
    model4.fit(X, Y)
    c4 = model4.predict(XDev) 
    accuracy = accuracy_score(YDev,c4)
    print("Accuracy for SVM:",accuracy * 100, " %")
    #cm = confusion_matrix(YTest, c4)
    #print("Confusion Matrix for SVM of Test data\n",cm)
    
    model6 = DecisionTreeClassifier(random_state=0)
    model6.fit(X, Y)
    c6 = model6.predict(XTest)
    #c6 = model6.predict(XDev)
    accuracy = accuracy_score(YTest,c6)
    print("Accuracy for C4.5:",accuracy * 100, " %")
    #cm = confusion_matrix(YTest, c6)
    #print("Confusion Matrix for C4.5 of Test data\n",cm)
    
    eclf1 = VotingClassifier(estimators=[('lr', model), ('rf', model4), ('gnb', model6)], voting='hard')
    eclf1 = eclf1.fit(X, Y)
    cVote1 = eclf1.predict(XTest) 
    accuracy = accuracy_score(YTest,cVote1)
    print("Accuracy for NB,C4.5 & SVM by Vote Ensemble:",accuracy * 100, " %")
    #cm = confusion_matrix(YTest, cVote1)
    #print("Confusion Matrix for NB,C4.5 & SVM by Vote Ensemble of Test data\n",cm)
    #==================DM Classifier With Voting Ensemble done here=======================#
    #=====================================================================================#
    
    
    #Training for SGD
    print("Training Accuracy Calculation for SGD")
    model2 = linear_model.SGDClassifier(loss="hinge", penalty="l2")
    model2.fit(X, Y)
    c2 = model2.predict(X)
    accuracy = accuracy_score(Y,c2)
    print("Accuracy for SGD(with (soft-margin) linear Support Vector Machine- loss parameter):",accuracy * 100, " %")
    #cm = confusion_matrix(Y, c2)
    #print("Confusion Matrix for SGD(with (soft-margin) linear Support Vector Machine- loss parameter of Test data\n",cm)
    
    model2 = linear_model.SGDClassifier(loss="modified_huber", penalty="l2")
    model2.fit(X, Y)
    c2 = model2.predict(X)
    accuracy = accuracy_score(Y,c2)
    print("Accuracy for SGD( smoothed hinge loss Parameter):",accuracy * 100, " %")
    
    model2 = linear_model.SGDClassifier(loss="log", penalty="l2")
    model2.fit(X, Y)
    c2 = model2.predict(X)
    accuracy = accuracy_score(Y,c2)
    print("Accuracy for SGD(logistic regression):",accuracy * 100, " %")
    
    model2 = linear_model.SGDClassifier(loss="modified_huber", alpha=0.5)
    model2.fit(X, Y)
    c2 = model2.predict(X)
    accuracy = accuracy_score(Y,c2)
    print("Accuracy for SGD(loss= modified_huber alpha=0.0001):",accuracy * 100, " %")
    
    
    #Validation for SGD
    print("Validation Accuracy Calculation for SGD")
    model2 = linear_model.SGDClassifier(loss="hinge", penalty="l2")
    model2.fit(X, Y)
    c2 = model2.predict(XDev)
    accuracy = accuracy_score(YDev,c2)
    print("Accuracy for SGD(with (soft-margin) linear Support Vector Machine- loss parameter):",accuracy * 100, " %")
    
    model2 = linear_model.SGDClassifier(loss="modified_huber", penalty="l2")
    model2.fit(X, Y)
    c2 = model2.predict(XDev)
    accuracy = accuracy_score(YDev,c2)
    print("Accuracy for SGD( smoothed hinge loss Parameter):",accuracy * 100, " %")
    
    model2 = linear_model.SGDClassifier(loss="log", penalty="l2")
    model2.fit(X, Y)
    c2 = model2.predict(XDev)
    accuracy = accuracy_score(YDev,c2)
    print("Accuracy for SGD(logistic regression):",accuracy * 100, " %")
    
    model2 = linear_model.SGDClassifier(loss="modified_huber", alpha=0.5)
    model2.fit(X, Y)
    c2 = model2.predict(XDev)
    accuracy = accuracy_score(YDev,c2)
    print("Accuracy for SGD(loss= modified_huber alpha=0.0001):",accuracy * 100, " %")

    
    
    #=========================================
    #Testing for SGD
    print("Testing Accuracy Calculation SGD")   
    model21 = linear_model.SGDClassifier(loss="hinge", penalty="l2")
    model21.fit(X, Y)
    c21 = model21.predict(XDev)
    accuracy = accuracy_score(YDev,c21)
    print("Accuracy for SGD(with (soft-margin) linear Support Vector Machine- loss parameter):",accuracy * 100, " %")
    
    model22 = linear_model.SGDClassifier(loss="modified_huber", penalty="l2")
    model22.fit(X, Y)
    c22 = model22.predict(XDev)
    accuracy = accuracy_score(YDev,c22)
    print("Accuracy for SGD(smoothed hinge loss Parameter):",accuracy * 100, " %")
    
    model23 = linear_model.SGDClassifier(loss="log", penalty="l2")
    model23.fit(X, Y)
    c23 = model23.predict(XDev)
    accuracy = accuracy_score(YDev,c23)
    print("Accuracy for SGD(logistic regression):",accuracy * 100, " %")
    
    eclf2 = VotingClassifier(estimators=[('lr', model21), ('rf', model22), ('gnb', model23)], voting='hard')
    eclf2 = eclf2.fit(X, Y)
    cVote2 = eclf2.predict(XDev) 
    accuracy = accuracy_score(YDev,cVote2)
    print("Accuracy for SGD(with (soft-margin),SGD( smoothed hinge loss Parameter) & SGD(logistic regression) by Vote Ensemble:",accuracy * 100, " %")
    #cm = confusion_matrix(YTest, cVote2)
    #print("Confusion Matrix for SGD(with soft-margin),SGD( smoothed hinge loss Parameter) & SGD(logistic regression) by Vote Ensemble of Test data\n",cm)

    #===================================
    
    
    #Training for MLP
    print("Training Accuracy Calculation") 
    model5 = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(4,100), activation='relu', random_state=1)
    model5.fit(X, Y)                         
    c5 = model5.predict(X) 
    accuracy = accuracy_score(Y,c5)
    print("Training Accuracy for MLP(‘lbfgs’ is an optimizer in the family of quasi-Newton methods):",accuracy * 100, "%")    
    
    model5 = MLPClassifier(solver='sgd', alpha=1e-5,hidden_layer_sizes=(4,100), activation='relu', random_state=1)
    model5.fit(X, Y)                         
    c5 = model5.predict(X) 
    accuracy = accuracy_score(Y,c5)
    print("Training Accuracy for MLP(refers to stochastic gradient descent):",accuracy * 100, "%")
    
    model5 = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(4,100), activation='relu', random_state=1)
    model5.fit(X, Y)                         
    c5 = model5.predict(X)
    accuracy = accuracy_score(Y,c5)
    print("Training Accuracy for MLP(stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba):",accuracy * 100, "%")
    
    
    #Validation for MLP
    print("\nValidation Accuracy Calculation") 
    model5 = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(4,100), activation='relu', random_state=1)
    model5.fit(X, Y)                         
    c5 = model5.predict(XDev) 
    accuracy = accuracy_score(YDev,c5)
    print("Validation Accuracy for MLP(‘lbfgs’ is an optimizer in the family of quasi-Newton methods):",accuracy * 100, "%")    
    
    model5 = MLPClassifier(solver='sgd', alpha=1e-5,hidden_layer_sizes=(4,100), activation='relu', random_state=1)
    model5.fit(X, Y)                         
    c5 = model5.predict(XDev) 
    accuracy = accuracy_score(YDev,c5)
    print("Validation Accuracy for MLP(refers to stochastic gradient descent):",accuracy * 100, "%")
    
    model5 = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(4,100), activation='relu', random_state=1)
    model5.fit(X, Y)                         
    c5 = model5.predict(XDev)
    accuracy = accuracy_score(YDev,c5)
    print("Validation Accuracy for MLP(stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba):",accuracy * 100, "%")
    
    
    #========================================
    #Testing for MLP
    print("\nTesting Accuracy Calculation of MLP") 
    model51 = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(4,100), activation='relu', random_state=1)
    model51.fit(X, Y)                         
    c51 = model51.predict(XTest) 
    accuracy = accuracy_score(YTest,c51)
    print("nTesting Accuracy for MLP(‘lbfgs’ is an optimizer in the family of quasi-Newton methods):",accuracy * 100, "%")    
    
    model52 = MLPClassifier(solver='sgd', alpha=1e-5,hidden_layer_sizes=(4,100), activation='relu', random_state=1)
    model52.fit(X, Y)                         
    c52 = model52.predict(XTest) 
    accuracy = accuracy_score(YTest,c52)
    print("nTesting Accuracy for MLP(refers to stochastic gradient descent):",accuracy * 100, "%")
    
    model53 = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(4,100), activation='relu', random_state=1)
    model53.fit(X, Y)                         
    c53 = model53.predict(XTest)
    accuracy = accuracy_score(YTest,c53)
    print("nTesting Accuracy for MLP(stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba):",accuracy * 100, "%")
    cm = confusion_matrix(YTest, c53)
    print("Confusion Matrix for stochastic gradient-based optimizer proposed by Kingma of Test data\n",cm)
    
    eclf3 = VotingClassifier(estimators=[('lr', model51), ('rf', model52), ('gnb', model53)], voting='hard')
    eclf3 = eclf3.fit(X, Y)
    cVote3 = eclf3.predict(XTest) 
    accuracy = accuracy_score(YTest,cVote3)
    print("Accuracy for MLP(quasi-Newton methods), MLP(SGD) & MLP(SGD by Kingma) by Vote Ensemble:",accuracy * 100, " %")
    #cm = confusion_matrix(YTest, cVote3)
    #print("Confusion Matrix for MLP(quasi-Newton methods),C4.5 & MLP(SGD by Kingma) by Vote Ensemble of Test data\n",cm)
    
    
    
    eclf4 = VotingClassifier(estimators=[('lr', eclf1), ('rf', eclf2), ('gnb', eclf3)], voting='hard')
    eclf4 = eclf4.fit(X, Y)
    cVote4 = eclf4.predict(XTest) 
    accuracy = accuracy_score(YTest,cVote4)
    print("Accuracy for Vote1, Vote2 & Vote3 by Vote Ensemble:",accuracy * 100, " %")
    #cm = confusion_matrix(YTest, cVote1)
    #print("Confusion Matrix for Vote1, Vote2 & Vote3 by Vote-Hard Ensemble of Test data\n",cm)
    #============================================================
    
    
    
    eclf4 = VotingClassifier(estimators=[('lr', eclf1), ('rf', eclf2), ('gnb', eclf3)], voting='soft')
    eclf4 = eclf4.fit(X, Y)
    cVote4 = eclf4.predict(XTest) 
    accuracy = accuracy_score(YTest,cVote4)
    print("Accuracy for Vote1, Vote2 & Vote3 by Vote Ensemble:",accuracy * 100, " %")
    #cm = confusion_matrix(YTest, cVote1)
    #print("Confusion Matrix for Vote1, Vote2 & Vote3 by Vote-Soft Ensemble of Test data\n",cm)
    
    
    eclf5 = VotingClassifier(estimators=[('lr', model6), ('rf', model23), ('gnb', model53)], voting='hard')
    eclf5 = eclf5.fit(X, Y)
    cVote5 = eclf5.predict(XDev) 
    accuracy = accuracy_score(YDev,cVote5)
    print("Accuracy for C4.5, SGD(with soft-margin) & MLP(SGD optimizer proposed by Kingma) by Vote Ensemble:",accuracy * 100, " %")
    #cm = confusion_matrix(YTest, cVote5)
    #print("Confusion Matrix for C4.5, SGD(Logistic regression) & MLP(SGD optimizer proposed by Kingma) by Vote Ensemble of Test data\n",cm)
    
if __name__=="__main__":
    main()