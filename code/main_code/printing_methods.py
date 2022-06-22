import numpy as np

from sklearn.linear_model import LinearRegression as linR
from sklearn.linear_model import LogisticRegression as logR
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn import svm

import general_methods


def print_importance(model, df = None, number = None):
    if isinstance(model, RandomForestClassifier):
        print_rf_importance(model, number)
    elif isinstance(model, logR):
        print_log_importance(model, number)
    elif isinstance(model, svm._classes.SVC):
        print_svm_importance(model, df, number)
    elif isinstance(model, XGBClassifier):
        print_xgb_importance(model, df, number)
    
def print_rf_importance(model, number = None):
    if number is None:
        number = len(model.feature_names_in_)
    y = model.feature_importances_
    x = model.feature_names_in_
    x = [general_methods.clean_title(n, wrap=False) for n in x]
    min_len = 0
    for i in x:
        if len(i) > min_len:
            min_len = len(i)
    
    feature_list = general_methods.merge_lists(x, y)
    sorted_feature_list = sorted(feature_list, key = lambda x: x[1], reverse = True)
    for enum, s in enumerate(sorted_feature_list[:number]):
        print("{:<3s}\t{:<{}}\t{:7.4f}".format(str(enum+1)+'.', s[0], min_len ,s[1]))
        
def print_log_importance(model, number = None):
    if number is None:
        number = 10
    
    x = model.feature_names_in_
    x = [general_methods.clean_title(n, wrap=False) for n in x]
    min_len = 0
    for i in x:
        if len(i) > min_len:
            min_len = len(i)
            
    for i, size in zip(range(len(model.classes_)), ['Small', 'Normal', 'Large']):
        y = model.coef_[i]
        feature_list = general_methods.merge_lists(x, y)
        sorted_feature_list = sorted(feature_list, key = lambda x: x[1], reverse = True)
        print("Model coefficients for {} predictor:".format(size))
        for enum, s in enumerate(sorted_feature_list[:number]):
            print("{:<3s}\t{:<{}}\t{:10.3e}".format(str(enum+1)+'.', s[0], min_len ,s[1]))
        
        print("\n")
        
def print_xgb_importance(model, df, number = None):
    if number is None:
        number = 10
        
    x = dfs[0].index.tolist()[:-2]
    x = [general_methods.clean_title(n, wrap=False) for n in x]
    shap_vals = get_shap_vals(model, df)

    min_len = 0
    for i in x:
        if len(i) > min_len:
            min_len = len(i)
            
    for i, size in zip(range(len(model.classes_)), ['Small', 'Normal', 'Large']):
        y = abs(np.mean(shap_vals[i], axis = 0))
        feature_list = general_methods.merge_lists(x, y)
        sorted_feature_list = sorted(feature_list, key = lambda x: x[1], reverse = True)
        print("Model coefficients for {} predictor:".format(size))
        for enum, s in enumerate(sorted_feature_list[:number]):
            print("{:<3s}\t{:<{}}\t{:10.3e}".format(str(enum+1)+'.', s[0], min_len ,s[1]))
        
        print("\n")
        
        
def print_svm_importance(model, df, number = None):
    if number is None:
        number = 10
        
    x = dfs[0].index.tolist()[:-2]
    x = [general_methods.clean_title(n, wrap=False) for n in x]
    shap_vals = get_shap_vals(model, df)

    min_len = 0
    for i in x:
        if len(i) > min_len:
            min_len = len(i)
            
    for i, size in zip(range(len(model.classes_)), ['Small', 'Normal', 'Large']):
        y = abs(np.mean(shap_vals[i], axis = 0))
        feature_list = general_methods.merge_lists(x, y)
        sorted_feature_list = sorted(feature_list, key = lambda x: x[1], reverse = True)
        print("Model coefficients for {} predictor:".format(size))
        for enum, s in enumerate(sorted_feature_list[:number]):
            print("{:<3s}\t{:<{}}\t{:10.3e}".format(str(enum+1)+'.', s[0], min_len ,s[1]))
        
        print("\n")


def summary_df(df):
    summary_dict = {'Mean':df.mean(axis=1), 'Std_Dev':df.std(axis=1),'Min':df.min(axis=1),
                    'Max':df.max(axis=1)}

    return pd.DataFrame.from_dict(summary_dict)

def get_metrics(pred, truth):
    if len(pred) != len(truth):
        return 0
    avg = metrics.accuracy_score(truth, pred)
    precision = metrics.precision_score(truth, pred, average = 'weighted')
    recall = metrics.recall_score(truth, pred, average = 'weighted')
    f1 =  metrics.f1_score(truth, pred, average = 'weighted')
    
    
    return {'avg':avg, 'precision':precision, 'recall':recall, 'f1':f1}
    