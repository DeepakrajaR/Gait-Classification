# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 13:09:42 2021

@author: Karansinh Padhiar
"""
# https://en.wikipedia.org/wiki/Sensitivity_and_specificity

# importing necessary packages
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.metrics import matthews_corrcoef
from numpy import sqrt

# Recall of Negative class [0]
def get_specificity(y_true, y_pred):
    return (recall_score(y_true=y_true, y_pred=y_pred, pos_label=0))
    # tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1]).ravel()
    # if tn + fp == 0:
    #     return (float(tn) / 1)
    # return (float(tn) / (tn + fp))

# recall of Positive class [1]
def get_sensitivity(y_true, y_pred):
    return (recall_score(y_true=y_true, y_pred=y_pred, pos_label=1))
    # tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1]).ravel()
    # if tp + fn == 0:
    #     return (float(tp) / 1)
    # return (float(tp) / (tp + fn))

def get_NPV(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1]).ravel()
    if tn + fn == 0:
        return (float(tn) / 1)
    return (float(tn) / (tn + fn))

def get_PPV(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1]).ravel()
    if tp + fp == 0:
        return (float(tp) / 1)
    return (float(tp) / (tp + fp))

def get_PLR(y_true, y_pred):
    if (1 - get_specificity(y_true, y_pred)) == 0:
        return (get_sensitivity(y_true, y_pred) / 1)
    return (get_sensitivity(y_true, y_pred) / (1 - get_specificity(y_true, y_pred)))

def get_MCC(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1]).ravel()
    numerator = (tp * tn) - (fp * fn)
    denominator = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        return (float(numerator) / 1)
    return (float(numerator) / denominator)
    # return (matthews_corrcoef(y_true=y_true, y_pred=y_pred))

