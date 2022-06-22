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
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from yellowbrick.classifier import ROCAUC

import general_methods



def plot_matrix(matrix, title = "Confusion Matrix", save_dir=None, disp = False, cmap = 'cool'):
    fig = plt.figure(facecolor = 'white', dpi = 400)
    ax1 = fig.add_subplot(111)
    # create heatmap
    ax1 = sns.heatmap(pd.DataFrame(matrix), annot=True, cmap=cmap ,fmt='g', ax = ax1, linewidths= 1)
    fig.suptitle(title, va = 'center',size = 20, weight = 'bold')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.xticks([0.5,1.5,2.5], ['Small', 'Normal', 'Large'])
    plt.yticks([0.5,1.5,2.5], ['Small', 'Normal', 'Large'])
    if save_dir is not None:
        title = title.replace('\n', '').replace(' ', '_')
        out_path = os.path.join(save_dir, title + '.png')
        plt.savefig(out_path, format = 'png', dpi = 300, bbox_inches = 'tight')
    if save_dir is None or disp: 
        plt.show()
    plt.close()

################################################ 
# Below are for plotting feature importance
# First method is wrapper to send to correct inner method
################################################ 

def plot_feature_importance(model, number = None, title = None, save_path = None, disp = False):
    if isinstance(model.mod, RandomForestClassifier):
        plot_feature_rf(model, number, title, save_path, disp)
    elif isinstance(model.mod, logR):
        plot_feature_logr(model, number, title, save_path, disp)
    elif isinstance(model.mod, svm._classes.SVC):
        plot_feature_svm(model, number, title, save_path, disp)
    elif isinstance(model.mod, XGBClassifier):
        plot_feature_xgb(model, number, title, save_path, disp)
        

def plot_feature_rf(model_obj, number = None, title = None, save_path = None, disp = False):
    model = model_obj.mod
    fig = plt.figure(facecolor = 'white', figsize = (15,5))
    ax = fig.add_subplot(111)
    x = model.feature_names_in_
    y = abs(model.feature_importances_)
    x = [general_methods.clean_title(n) for n in x]
    
    if number is None or number == 0:
        number = len(x)
    merge_l = general_methods.merge_lists(x,y)
    merge_l.sort(key = lambda x: x[1], reverse = True)
    merge_l = merge_l[:number]
    x_sort, y_sort = zip(*merge_l)
    
    #ax.set_xticks([x for x in range(len(y))])
    ax.set_xticklabels(x_sort, ha = 'right', rotation = 45)
    ax.set_ylim([np.min(y_sort), np.max(y_sort)*1.05])
    #print_mod_order(x_sort, y_sort)
    ax = plt.bar(x_sort, y_sort)
    if title is not None:
        title = title.replace('_', ' ')
        fig.suptitle(title.title(), va = 'center',size = 20, weight = 'bold')
    
    if save_path is not None:
        if title is None:
            title = model_obj.run_name
        out_file = os.path.join(save_path, title.replace(' ', '_') + '.png')
        plt.savefig(out_file, format = 'png', dpi = 300, bbox_inches = 'tight')
    if save_path is None or disp:
        plt.show()
    plt.close()

    
    
def plot_feature_logr(model_obj, number = None, title = None, save_path = None, disp = False):
    model = model_obj.mod
    fig = plt.figure(facecolor = 'white', figsize = (10,9))
    
    for i, size in zip(range(len(model.classes_)), ['Small', 'Normal', 'Large']):
        ax = fig.add_subplot(3, 1, i+1)
        y = abs(model.coef_[i])
        x = model.feature_names_in_
        #x = [general_methods.clean_title(n) for n in x]
        x = [item.replace(model_obj.structure,'').replace('_', ' ').strip().title() for item in x]
        x = general_methods.clean_feature_names(x)
    
        if number is None or number == 0:
            number = len(x)
        merge_l = general_methods.merge_lists(x,y)
        merge_l.sort(key = lambda x: x[1], reverse = True)
        merge_l = merge_l[:number]
        x_sort, y_sort = zip(*merge_l)
        #ax.tick_params(axis='x', labelrotation = 70)
        
        ax.set_xticklabels(x_sort, ha = 'right', va = 'top', rotation = 45)
        ax.set_ylim([np.min(y_sort), np.max(y_sort)*1.05])
        #print_mod_order(x_sort, y_sort)
        ax.bar(x_sort, y_sort)
        for t in ax.get_yticklabels():
            print(t)
        #ticks = color_labels(ax._axes.get_yticklabels())
        #ax.set_yticks(ticks)
        
    if title is not None:
        title = title.replace('_', ' ')
        fig.suptitle(title.title(), va = 'center',size = 20, weight = 'bold')

    plt.subplots_adjust(hspace = 1.5)
    fig.align_labels()
    
    if save_path is not None:
        if title is None:
            title = model_obj.run_name
        out_file = os.path.join(save_path, title.replace(' ', '_') + '.png')
        plt.savefig(out_file, format = 'png', dpi = 300, bbox_inches = 'tight')
    if save_path is None or disp:
        plt.show()
    plt.close()
    return t
    

def plot_feature_xgb(model_obj, number = None, title = None, save_path = None, disp = False):  
    model = model_obj.mod
    df = model_obj.df
    fig = plt.figure(facecolor = 'white', figsize = (15,12))
    
    for i, size in zip(range(len(model.classes_)), ['Small', 'Normal', 'Large']):
        ax = fig.add_subplot(3, 1, i+1)
        x = df.index.tolist()[:-2]
        #abs(np.mean(shap_values[1], axis = 0))
        shap_vals = general_methods.get_shap_vals(model_obj)
        y = abs(np.mean(shap_vals[i], axis = 0))
        x = [general_methods.clean_title(n) for n in x]
             
        if number is None or number == 0:
            number = len(x)
        merge_l = general_methods.merge_lists(x,y)
        merge_l.sort(key = lambda x: x[1], reverse = True)
        merge_l = merge_l[:number]
        x_sort, y_sort = zip(*merge_l)
        #ax.tick_params(axis='x', labelrotation = 70)
        ax.set_xticklabels(x_sort, ha = 'right', rotation = 45)
        ax.set_ylim([np.min(y_sort), np.max(y_sort)*1.05])
        print_mod_order(x_sort, y_sort)
        ax = plt.bar(x_sort, y_sort)
        
    if title is None:
        title = "Place Holder"
    else:
        title = title.replace('_', ' ')
    fig.suptitle(title.title(), va = 'center',size = 20, weight = 'bold')
    plt.subplots_adjust(hspace = 1.5)
    fig.align_labels()
    
    if save_path is not None:
        out_file = os.path.join(save_path, title.replace(' ', '_') + '.png')
        plt.savefig(out_file, format = 'png', dpi = 300, bbox_inches = 'tight')
    if save_path is None or disp:
        plt.show() 
    plt.close()
        
def plot_feature_svm(model_obj, number = None, title = None, save_path = None, disp = False):  
    model = model_obj.mod
    df = model_obj.df
    fig = plt.figure(facecolor = 'white', figsize = (15,12))
    
    for i, size in zip(range(len(model.classes_)), ['Small', 'Normal', 'Large']):
        ax = fig.add_subplot(3, 1, i+1)
        x = df.index.tolist()[:-2]
        #abs(np.mean(shap_values[1], axis = 0))
        shap_vals = general_methods.get_shap_vals(model_obj)
        y = abs(np.mean(shap_vals[i], axis = 0))
        x = [general_methods.clean_title(n) for n in x]
             
        if number is None or number == 0:
            number = len(x)
        merge_l = general_methods.merge_lists(x,y)
        merge_l.sort(key = lambda x: x[1], reverse = True)
        merge_l = merge_l[:number]
        x_sort, y_sort = zip(*merge_l)
        #ax.tick_params(axis='x', labelrotation = 70)
        ax.set_xticklabels(x_sort, ha = 'right', rotation = 45)
        ax.set_ylim([np.min(y_sort), np.max(y_sort)*1.05])
        print_mod_order(x_sort, y_sort)
        ax = plt.bar(x_sort, y_sort)
        
    if title is None:
        title = "Place Holder"
    else:
        title = title.replace('_', ' ')
    fig.suptitle(title.title(), va = 'center',size = 20, weight = 'bold')
    plt.subplots_adjust(hspace = 1.5)
    fig.align_labels()
    
    if save_path is not None:
        out_file = os.path.join(save_path, title.replace(' ', '_') + '.png')
        plt.savefig(out_file, format = 'png', dpi = 300, bbox_inches = 'tight')
    if save_path is None or disp:
        plt.show() 
    plt.close()

def plot_roc(model, df, title = None, save_path = None, disp = False):
    x = df.iloc[:-2,:].T
    y = df.iloc[-1,:]
    
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.8) 
    y_prob_pred_cnb = model.predict_proba(test_x)

    # roc curve for classes
    fpr = {}
    tpr = {}
    thresh = {}
    
    mod_auroc = roc_auc_score(test_y, y_prob_pred_cnb, multi_class='ovo', average='weighted')
    n_class = 3
    fig = plt.figure(facecolor = 'white', figsize = (5,5), dpi = 300)
    ax = fig.add_subplot(111)
    for i in range(n_class):    
        fpr[i], tpr[i], thresh[i] = roc_curve(test_y, y_prob_pred_cnb[:,i], pos_label=i)

    # plotting
    thick = 0.6
    ax = plt.plot(fpr[0], tpr[0], linestyle='--', linewidth = thick, color='red', label='Small vs Rest')
    ax = plt.plot(fpr[1], tpr[1], linestyle='--', linewidth = thick, color='grey', label='Normal vs Rest')
    ax = plt.plot(fpr[2], tpr[2], linestyle='--', linewidth = thick, color='blue', label='Large vs Rest')
    text_dict = {'fontsize': 5,}
    #ax = plt.text(0.5, 0.99, 'Weighted AUROC: {:.3f}'.format(mod_auroc), text_dict)
    plt.title('Weighted AUROC: {:.3f}'.format(mod_auroc), fontsize = 12)
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    
    if title is None:
        title = "Multiclass ROC curve"
    else:
        title = title.replace('_', ' ')
    fig.suptitle(title.title(), va = 'center',size = 20, weight = 'bold')
    
    if save_path is not None:
        out_file = os.path.join(save_path, title.replace(' ', '_') + '.png')
        plt.savefig(out_file, format = 'png', dpi = 300, bbox_inches = 'tight')
    if save_path is None or disp:
        plt.show()
    plt.close()


def plot_yellowbrick_roc(model, df, title, save_path = None, disp = False):
    X = df.iloc[:-2,:].T
    y = df.iloc[-1,:]
    
    y = LabelEncoder().fit_transform(y)
    
    # Create the train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Instaniate the classification model and visualizer
    fig = plt.figure(facecolor = 'white', figsize = (7,5), dpi = 300)
    
    ax = fig.add_subplot(111)
    title = general_methods.clean_title(title, wrap = True)
    fig.suptitle(title.title(), va = 'center',size = 20, weight = 'bold')
    visualizer = ROCAUC(model, classes=["Small", "Normal", "Large"], ax = ax)

    visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)        # Evaluate the model on the test data
    plt.legend()
    
    if save_path is not None:
        out_file = os.path.join(save_path, title.replace(' ', '_').replace('\n','') + '.png')
        plt.savefig(out_file, format = 'png', dpi = 300, bbox_inches = 'tight')
    if save_path is None or disp:
        visualizer.show()
        plt.show()    
    plt.close()

    
def plot_yellowbrick_roc_xy(model, x, y, title, save_path = None, disp = False):
    

    # Instaniate the classification model and visualizer
    fig = plt.figure(facecolor = 'white', figsize = (7,5), dpi = 300)
    
    ax = fig.add_subplot(111)
    title = general_methods.clean_title(title, wrap = True)
    fig.suptitle(title.title(), va = 'center',size = 20, weight = 'bold')
    visualizer = ROCAUC(model, classes=["Small", "Normal", "Large"], ax = ax)

    visualizer.fit(x, y)        # Fit the training data to the visualizer
    visualizer.score(x, y)       # Evaluate the model on the test data
    plt.legend()
    
    if save_path is not None:
        out_file = os.path.join(save_path, title.replace(' ', '_').replace('\n','') + '.png')
        visualizer.show(outpath = out_file)
    if save_path is None or disp:
        visualizer.show()
        plt.show()    
    plt.close()


def print_mod_order(x_sort, y_sort):
    for enum, (a, b) in enumerate(zip(x_sort,y_sort)):
        print("{:3d}: {:<40s} {:.3f}".format(enum+1, general_methods.clean_title(a, wrap = False), b))
    print()
    
    
def color_labels(ticks):
    color_dict = {'glrlm':'green', 'firstorder':'red', 'glcm':'blue', 'shape':'black'}
    for item in ticks:
        text = item.__dict__['_text']
        for key in color_dict.keys():
            if key in text.lower():
                item._color = color_dict[key]
                break

    return ticks