import random
import sys
import pandas as pd
import os, sys, pickle
import shap
from sklearn.model_selection import train_test_split
import textwrap

import locations
from model_class import model



def generate_rand_list(length, maxn, seed = None):
    if seed is not None:
        random.seed(seed)
    randl = []
    
    while len(randl) < length:
        new_rand = random.randrange(0, maxn)
        if new_rand not in randl:
            randl.append(new_rand)
    return randl

def merge_lists(*kargs):
    a_len = len(kargs[0])
    for arg in kargs:
        if len(arg) != a_len:
            return None
        
    flist = []
    for i in range(a_len):
        temp = []
        for arg in kargs:
            temp.append(arg[i])
        flist.append(tuple(temp))
    return flist


def return_phrases():
    phrases = ['Deep_grey', 'Brain_stem', 'Left_wm', 'Left_gm', 
           'Right_wm', 'Right_gm', 'Left_cerebellum', 'Right_cerebellum']

    return phrases

  
    
def get_shap_vals(model_obj: model):
    if model_obj.shap_value is None:
        df = model_obj.df
        mod = model_obj.mod
        x = df.iloc[:-2,0:].T
        y = df.iloc[-1,:] 
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.80, random_state = 1)
        explainer = shap.KernelExplainer(mod.predict_proba, x_train)
        small_x = shap.sample(x_test, 50)
        model_obj.shap_value = explainer.shap_values(small_x)
        return model_obj.shap_value
    else:
        return model_obj.shap_value

def get_shap_vals2(model_obj: model):
    if model_obj.shap_value is None:
        df = model_obj.df
        mod = model_obj.mod
        x = df.iloc[:-2,0:].T
        y = df.iloc[-1,:] 
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.80, random_state = 1)
        explainer = shap.KernelExplainer(mod.predict_proba, x_train)
        small_x = shap.sample(x_test, 50)
        model_obj.shap_value = explainer.shap_values(small_x)
        return model_obj.shap_value
    else:
        return model_obj.shap_value


def save_mod_results(model_result_list):
    save_dir = locations.get_locations('save_dir')
    os.makedirs(save_dir, exist_ok = True)
    save_date_string = locations.get_locations('save_date')
    save_path = os.path.join(save_dir, '{}_hashed_models.pkl'.format(save_date_string))
    try:
        with open(save_path, 'wb') as f:
            pickle.dump(model_result_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        return True
    except Exception as e:
        print(e)
        return False
        
        
def load_mod_results_check(list_length) -> bool:
    try:
        hashed_mods_pkl = locations.get_locations('hashed_mods')
        with open(hashed_mods_pkl, 'rb') as f:
            model_result_list = pickle.load(f)
            if len(model_result_list) != list_length:
                msg = "Model result list is not the correct length, missing models"
                msg = msg + "\nExpected {} but loaded model list only contained {}".format(list_length, len(model_result_list))
                raise Exception(msg)
        return True
    except Exception as e:
        print(e)
        return False
     
def load_mod_results() -> list:
    try:
        save_dir = locations.get_locations('save_dir')
        load_date_string = locations.get_locations('load_date')
        load_path = os.path.join(save_dir, '{}_hashed_models.pkl'.format(load_date_string))
        with open(load_path, 'rb') as f:
            model_result_list = pickle.load(f)
        return model_result_list 
    except Exception as e:
        print("Import failed due to {}".format(str(e)))
        return None


def clean_title(title, wrap = True):
    title = title.replace('\n', '')
    s_l = title.split('_')
    ind = 0
    for enum, s in enumerate(s_l):
        if s == 'original':
            ind = enum + 1
            break
            
    if s_l[ind] == 'firstorder':
        s_l[ind] = 'FO'
    if wrap:
        return textwrap.fill(' '.join(s_l[ind:]), 20)
    else:
        return ' '.join(s_l[ind:])



def clean_feature_names(feature_list, left=False) -> list:
    remove_list = ['Original Shape', 'Original Firstorder', 'Original Glcm', 'Original Glrlm']
    remove_list = [x.lower() for x in remove_list]
    
    find_l = ['run', 'variance', 'length', 'level', 'high', 'low', 'mean', 'absolute', 
              'percentile', 'range', 'priminence', 'shade', 'tendency', 'average', 'entropy',
             'probability', 'normalized']
    new_list = []
    for item in feature_list:
        item = item.lower()
        for rem in remove_list:
            item = item.replace(rem, '')
        
        for find in find_l:
            repl = f' {find} '
            item = item.replace(find, repl)
        
        
        
        if len(item) < 6:
            item = item.upper()
        else:
            item = item.title()
        item = " ".join(item.split())
        new_list.append(item.strip())
    if left:
        lens = [len(x) for x in new_list]
        max_len = max(lens)
        new_list = [x.ljust(max_len, ' ') for x in new_list]
        return new_list
    else:
        return new_list