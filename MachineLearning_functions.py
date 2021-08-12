# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 08:28:57 2021

@author: phil
"""

## This function is used to produce machine learning performance metrics
## for both crime and twitter classifiers

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

def RF_performance_reporting(rf, test_features, test_labels, y_pred, description, print_output=True):    
    # I now want to create a confusion matrix and to do this I first have to 
    # change my model output from a probability to a label 1 or 0
   
    accuracy = accuracy_score(test_labels, y_pred)
    precision = precision_score(test_labels, y_pred)
    recall = recall_score(test_labels, y_pred)
    roc_auc = roc_auc_score(test_labels, y_pred)
    cm = confusion_matrix(test_labels, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(test_labels,y_pred).ravel()
    
    if print_output:
        
        print('------------------------------------------------------------------')
        print('Printing performance metrics for ' + description)
        print('------------------------------------------------------------------\n')
        print('prediction accuracy', accuracy)
        print('precision', precision)
        print('recall', recall)
        print('Area under the curve score', roc_auc)   
        print('\n')  

        fpr, tpr, thresholds = roc_curve(test_labels, y_pred)

        plot_confusion_matrix(rf, test_features, test_labels)  
        plt.show() 
        
        metrics.plot_roc_curve(rf, test_features, test_labels)  
        plt.show()
        
    return accuracy, precision, recall, roc_auc, tn, fp, fn, tp