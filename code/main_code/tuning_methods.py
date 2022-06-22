import os, sys, time, psutil
from datetime import datetime
import numpy as np

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV, RandomizedSearchCV
import multiprocessing


def get_df(df, phrase):
    keep_cols = []
    for col_name, data in df.items():
        if phrase.lower() in col_name.lower():
            keep_cols.append(col_name)
    volume_col = f'Volume_{phrase}'
    return df.loc[:,keep_cols].drop(volume_col, axis = 0).copy()

def get_total_mods(search_params):
    total = 0
    if isinstance(search_params, dict):
        total+=get_dict_combos(search_params)
    else:
        for param_dict in search_params:
            total+=get_dict_combos(param_dict)
    return total

def get_dict_combos(param_dict):
    total = 1
    for v in param_dict.values():
        total*=len(v)
    return total

def calc_cores(threshold = 19):
    core_usage = psutil.cpu_percent(interval=0, percpu=True)
    avail_cores = len([x for x in core_usage if x < threshold])
    max_cores = .75 * multiprocessing.cpu_count()
    run_cores = int(min(avail_cores, max_cores))
    return run_cores

def tune_hyperparams(x, y, mod, search_params, verb = -1, run_cores = None, cv_num = 10):
    if run_cores is None:
        run_cores = calc_cores()
    start_time = time.time()
    total_mods = get_total_mods(search_params)
    print(f'Total model combinations: {total_mods} using {run_cores} cores')
    clf = GridSearchCV(mod, search_params, n_jobs = run_cores, cv = cv_num, verbose = verb, scoring = 'roc_auc_ovr_weighted')
    clf.fit(x, y)
    end_time = time.time()
    return clf

def tune_hyperparams_random(x, y, mod, search_params, verb = -1, run_cores = None, cv = 10, n_iter = 50):
    if run_cores is None:
        run_cores = calc_cores()
    start_time = time.time()
    total_mods = get_total_mods(search_params)
    print(f'Total model combinations: {total_mods} using {run_cores} cores')
    clf = RandomizedSearchCV(mod, search_params, n_jobs = run_cores, n_iter = n_iter, cv = cv, verbose = verb, scoring = 'roc_auc_ovr_weighted')
    clf.fit(x, y)
    end_time = time.time()
    return clf

def build_log_params(print_size = False):
    tol_params = np.linspace(0e-5, 1e-3, 5).tolist()
    C_params = np.linspace(1, 5, 3).tolist()
    fit_intercept_params = [True, False]
    max_iter_params = np.arange(99, 1000, 100).tolist()
    l1_ratio_params = np.linspace(0,1, 5).tolist()


    logr_search_params = [{
        'penalty': ['elasticnet'],
        'tol': tol_params,
        'C': C_params, 
        'fit_intercept': fit_intercept_params, 
        'max_iter': max_iter_params, 
        'l1_ratio': l1_ratio_params
    }, 
    {
        'penalty': ['l1', 'l2'],
        'tol': tol_params,
        'C': C_params, 
        'fit_intercept': fit_intercept_params, 
        'max_iter': max_iter_params
    }]

    if print_size:
        log_total = get_total_mods(logr_search_params)
        print(f'Total model combinations: {log_total}')

    return logr_search_params


def build_rf_params(print_size = False):

    n_estimators_params = np.arange(99, 1000, 100).astype(int).tolist()
    criterion_params = ['gini', 'entropy']
    max_depth_params = [None] + np.arange(4, 30, 5).astype(int).tolist()
    max_features_params = ['sqrt', 'log2'] + np.arange(10, 50, 10).astype(int).tolist()
    warm_start_params = [True, False]
    max_leaf_nodes_params =  [None] + list(np.arange(9, 50, 10).astype(int))
    min_samples_split_params = [2, 5, 10]
    

    rf_search_params = [{
        'n_estimators':n_estimators_params, 
        'criterion': criterion_params,
        'max_depth': max_depth_params, 
        'max_features':max_features_params,
        'warm_start':warm_start_params,
        'max_leaf_nodes': max_leaf_nodes_params,
        'min_samples_split': min_samples_split_params,
    }]


    if print_size:
        rf_total = get_total_mods(rf_search_params)
        print(f'Total model combinations: {rf_total}')
    return rf_search_params



def build_svm_params(print_size = False):
    C_params = [0.1, 1, 2]
    shrinking_params = [True, False]
    tol_params = np.linspace(0e-5, 1e-3, 5).tolist()

    svm_search_params = [{
        'kernel':['poly'],
        'degree': [2, 4, 5], 
        'C': C_params,
        'shrinking': shrinking_params,
        'tol': tol_params,
    },
    {
        'kernel': ['rbf'],
        'C': C_params,
        'shrinking': shrinking_params,
        'tol': tol_params,
            
    },
    {
        'kernel': ['rbf', 'sigmoid'],
        'coef0': [0, 0.5, 1],
        'C': C_params,
        'shrinking': shrinking_params,
        'tol': tol_params,
            
    }]


    if print_size:
        svm_total = get_total_mods(svm_search_params)
        print(f'Total model combinations: {svm_total}')
    return svm_search_params


def build_xgb_params(print_size = False):
    # XGB Classifier Search Params
    xgb_search_params = [{
        'min_child_weight': [0, 5, 10],
        'gamma': [1, 1.5, 2, 5],
        'subsample': np.linspace(0, 1, 5).tolist(), 
        'colsample_bytree': np.linspace(0, 1, 5).tolist(), 
        'max_depth': [2, 4, 5]
    }]

    if print_size:
        xgb_total = get_total_mods(xgb_search_params)
        print(f'Total model combinations: {xgb_total}')
    return xgb_search_params


def current_time(ret = False, disp = True):
    now = datetime.now()
    print_time = now.strftime('%Y-%m-%d -- %H:%M:%S')
    if disp:
        print(print_time)
    if ret:
        return print_time