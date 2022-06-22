import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
import pandas as pd
import numpy as np
import textwrap

from sklearn.linear_model import LinearRegression as linR
from sklearn.linear_model import LogisticRegression as logR
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn import svm

import general_methods
from model_class import model, model_results


def get_feature_importance(model, number = 10):
    if isinstance(model.mod, RandomForestClassifier):
        return rf_feature_importance(model, number)
    elif isinstance(model.mod, logR):
        return logr_feature_importance(model, number)
    elif isinstance(model.mod, svm._classes.SVC):
        return svm_feature_importance(model, number)
    elif isinstance(model.mod, XGBClassifier):
        return xgb_feature_importance(model, number)

    


def rf_feature_importance(model, number):
    return abs(model.mod.feature_importances_)

def logr_feature_importance(model, number):
    r_list = []
    for i in range(3):
        r_list.append(abs(model.mod.coef_[i]))
    return r_list


def svm_feature_importance(model, number):
    r_list = []
    shap = general_methods.get_shap_vals(model)
    for i in range(3):
        r_list.append(abs(np.mean(shap[i], axis = 0)))
    return r_list

def xgb_feature_importance(model, number):
    r_list = []
    for i in range(3):
        r_list.append(abs(model.feature_importances_[i]))
    return r_list