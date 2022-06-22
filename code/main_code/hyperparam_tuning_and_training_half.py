import os, sys
import time
import pandas as pd
import numpy as np
import pickle

# machine leanring methods
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# my methods 
import locations
import general_methods
import df_methods

# my methods for tuning
import tuning_methods as tm




def main(index):
    data_loc = locations.get_locations('data_csv_loc')
    data = pd.read_csv(data_loc, sep=',', header = 0, index_col = 0)
    scaled_data = df_methods.scale_df(data)  
    dfs, sdfs = df_methods.setup_data(data, scaled_data)
    bdfs, sbdfs = df_methods.balance_data(dfs, sdfs)
    phrases = general_methods.return_phrases()

    cv_seed = 1234


    model_starts = [
        RandomForestClassifier(bootstrap = True),
        LogisticRegression(multi_class='multinomial', warm_start = True, solver = 'saga'),
        svm.SVC(probability = True),
        XGBClassifier(use_label_encoder=False, eval_metric = 'mlogloss')
    ]

    model_names = ['rf', 'logr', 'svm', 'xgb']
    search_params = [tm.build_rf_params(), tm.build_log_params(), tm.build_svm_params(), tm.build_xgb_params()]

    # Setting up save locations
    save_dir = locations.get_locations('save_dir') 
    hyper_dir = os.path.join(save_dir, 'hyper_training')
    os.makedirs(hyper_dir, exist_ok = True)


    merge_df_lists = [sbdfs, bdfs]
    df_types = ['Scaled_balanced', 'balanced']

    df_list = merge_df_lists[index]
    df_type = df_types[index]

    pickle_clf_list_loc = os.path.join(hyper_dir, f'{df_type}_hyper_training.pkl')
    pickle_clf_dict_loc = os.path.join(hyper_dir, f'{df_type}_dict_hyper_training.pkl')
    text_results_loc = os.path.join(hyper_dir, f'{df_type}_hyper_training_status.txt')
    failed_runs_loc = os.path.join(hyper_dir, f'{df_type}_failed_model_attempts.txt')
    
    result_tups = []
    if os.path.isfile(pickle_clf_dict_loc):
        with open(pickle_clf_dict_loc, 'rb') as f:
            result_dict = pickle.load(f)
    else:
        result_dict = {}
    

    total_models = 8 * 4
    current_model_count = 0

    #for df_list, df_type in zip(merge_df_lists, df_types):
    
    all_start_time = time.time()
    
    for df, phrase in zip(df_list, phrases):
        df = df.iloc[:, :].T
        x,y = df.iloc[:, :-2], df.iloc[:, -1]
        y = y+1 #xgb needs positive class values
        train_x, test_x, train_y, test_y = train_test_split(x,y, train_size = 0.8, random_state = cv_seed)

        for model, model_method, params in zip(model_starts, model_names, search_params):
            current_model_count+=1
            model_start_time = time.time()
            model_name_string = f'{df_type}_{phrase}_{model_method}'
            if model_name_string not in result_dict:
                try:
                    print(f'Running {model_name_string}')
                    tm.current_time()
                    new_model = model
                    clf = tm.tune_hyperparams_random(train_x, train_y, new_model, params, verb = 1, run_cores = 1, cv = 10, n_iter = 100)
                    runtime = time.time() - model_start_time

                    res_tuple = (df_type, phrase, model_method, clf, runtime, test_x, test_y)
                    result_tups.append(res_tuple)
                    result_dict[model_name_string] = res_tuple
                    
                    current_runtime = time.time() - all_start_time
                    res_string = f'{current_model_count}/{total_models}: {df_type}_{phrase}_{model_method}_{runtime:0.3f}_{current_runtime:0.3f}\n'

                    with open(text_results_loc, 'a') as f:
                        f.write(res_string)
                    with open(pickle_clf_list_loc, 'wb') as f:
                        pickle.dump(result_tups, f)
                    with open(pickle_clf_dict_loc, 'wb') as f:
                        pickle.dump(result_dict, f)


                except Exception as e:
                    print(e)

                    current_runtime = time.time() - all_start_time
                    runtime = time.time() - model_start_time
                    res_string = f'{current_model_count}/{total_models}: {df_type}_{phrase}_{model_method}_{runtime:0.3f}_{current_runtime:0.3f}\n'
                    with open(failed_runs_loc, 'a') as f:
                        f.write(res_string)
                        f.write(str(e))

if __name__ == '__main__':
    if len(sys.argv) > 1:
        try:
            index = int(sys.argv[1])
            main(index)
        except Exception as e:
            print(e)
            pass
    else:
        print("Need input of 0 or 1")