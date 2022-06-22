import os
from re import I
import pandas as pd

class model:
    def __init__(self, mod_object = None):
        self.mod = None
        self.accuracy = None
        self.f1 = None
        self.precision = None
        self.recall = None
        self.avg_score = None
        self.auroc = None
        self.df = None
        self.df_name = None
        self.run_name = None
        self.shap_value = None
        self.structure = None
        self.method = None
        self.model_id = None
        self.model_features = None
        
        if mod_object is not None:
            self.set_features(mod_object)

    def set_features(self, mobj):
        self_dict = self.__dict__
        for key in self.__dict__.keys():
            if key in mobj.__dict__:
                self_dict[key] = mobj.__dict__[key]
            else:
                self_dict[key] = None


    def print_results2(self, label_len, num = None):
        self.update_run_name()
        if num is None:
            print("{:<{}s} Accuracy: {:>4.2f}, F1: {:>4.2f}, Precision: {:>4.2f}, Recall: {:>4.2f}, Avg Score: {:>4.2f} ModelID: {:>3d}".format(self.run_name, 
                label_len, self.accuracy,self.precision, self.f1, self.recall, self.avg_score, self.model_id))
        else:
            print("{:>3d}: {:<{}s} Accuracy: {:>4.2f}, F1: {:>4.2f}, Precision: {:>4.2f}, Recall: {:>4.2f}, Avg Score: {:>4.2f}, ModelID: {:>3d}".format(num, self.run_name,
                label_len, self.accuracy,self.precision, self.f1, self.recall, self.avg_score, self.model_id))

    def print_results(self, num = None):
        skip_params = ['run_name', 'df', 'shap_value', 'feature_values', 'model_features', 'mod']
        dict_len = len(self.__dict__)
        first = True
        for enum, (k,v) in enumerate(self.__dict__.items()):
            if k in skip_params:
                continue
            if not first and v is not None:
                print(', ', end = '')
            first = False
            if isinstance(v, str):
                print("{}: {}".format(k.title(), v), end = '')
            elif isinstance(v, float):
                print("{}: {:>4.3f}".format(k.title(), v), end = '')
            elif isinstance(v, int):
                print("{}: {:d}".format(k.title(), v), end = '')

    def calc_avg_score(self):
        self.avg_score = (self.accuracy + self.f1 + self.precision + self.recall)/4

    def update_run_name(self):
        self.run_name = self.run_name.replace('\n','')
    
    def update_structure_name(self, phrase):
        self.structure = phrase

    def update_method(self, method):
        self.method = method

    def get_results(self):
        return (self.run_name, self.method, self.df_name, self.structure, 
                self.accuracy, self.f1, self.precision, self.recall)

    def print_dict(self):
        exclude_list = ['df', 'mod', 'shap_value']
        for k,v in self.__dict__.items():
            if k in exclude_list:
                continue
            print("{:>20s}: {}".format(k,v))

    def get_features(self):
        self.model_features = self.df.index.tolist()[:-1]
        """
        def print_feature_importance(self, number = 10):
            feature_importance = model_methods.get_feature_importanace(self)
            if self.model_features is None:
                self.get_features()
        """

        #if len(feature_importance) == 1:
        if isinstance(feature_importance[1], float):
            merge_list = [(a,b) for a,b in zip(feature_importance, self.model_features)]
            merge_list.sort(key = lambda x: x[0], reverse = True)
            val, name = zip(*merge_list)
            for enum, i in enumerate(range(number)):
                print("{:>3d} {:>5.3f} {:>15s}".format(enum+1, val[i], name[i]))
        else:
            for i, size_name in zip([0,1,2], ['Small', 'Normal', 'Large']):
                print(size_name)
                merge_list = [(a,b) for a,b in zip(feature_importance[i], self.model_features)]
                merge_list.sort(key = lambda x: x[0], reverse = True)
                val, name = zip(*merge_list)
                for enum, j in enumerate(range(number)):
                    print("{:>3d} {:>5.3f} {:>15s}".format(enum+1, val[j], name[j]))
                print()

        print()

    def __str__(self):
        return str(self.run_name)


    def toSeries(self):
        skip_params = ['df', 'shap_value', 'feature_values', 'model_features', 'mod']
        s = pd.Series([], dtype = 'object') 
        for enum, (k,v) in enumerate(self.__dict__.items()):
            if k in skip_params:
                continue
            s_temp = pd.Series([v], index = [k])
            s = s.append(s_temp)
        return s


class model_results:
    def __init__(self, models=[]):
        self.model_list = []
        self.__number_mods = 0
        self.__longest_phrase = 0
        self.__index = 0
        if models is not None and len(models) > 0:
            self.set_list(models)
            self.update_list_info()

    def update_list_info(self):
        self.__number_mods = len(self.model_list)
        self.__longest_phrase = self.get_max_phrase_length()

    #######################################
    # Get methods
    def get_max_phrase_length(self) -> int:
        max_len = 0
        for mod in self.model_list:
            if len(mod.run_name) > max_len:
                max_len = len(mod.run_name)
        return max_len
    
    def get_list(self):
        return self.model_list
    
    #######################################
    # Set methods 
    def set_list(self, model_list):
        self.model_list = model_list
        self.update_list_info()
    
    def set_longest_phrase(self, val: int):
        self.__longest_phrase == val

    
    #######################################
    ## normal methods
    def add_model(self, model_obj):
        self.model_list.append(model_obj)
        self.__number_mods+=1
        if len(model_obj.run_name) > self.__longest_phrase:
            self.__longest_phrase = len(model_obj.run_name)

    def sort_results(self, attribute = 'accuracy', print_res = True):
        if attribute == 'accuracy':
            self.model_list.sort(key = lambda x: x.accuracy, reverse=True)
        elif attribute == 'f1':
            self.model_list.sort(key = lambda x: x.f1, reverse=True)
        elif attribute == 'precision':
            self.model_list.sort(key = lambda x: x.precision, reverse=True)
        elif attribute == 'recall':
            self.model_list.sort(key = lambda x: x.recall, reverse=True)
        elif attribute == 'avg':
            self.model_list.sort(key = lambda x: x.avg_score, reverse=True)
        if print_res:
            self.print_results()
    
    def join_list(self, mod_r):
        self.model_list.extend(mod_r.get_list())
        self.update_list_info()
        return self


    def print_results2(self, number = None):
        if number == None:
            number = self.__number_mods
        for enum, model_obj in enumerate(self.model_list[:number]):
            model_obj.print_results2(self.__longest_phrase, enum+1)
        print()

    def print_results(self, number = None):
        if number is None:
            number = self.__number_mods
        for mobj in self.model_list[:number]:
            mobj.print_results()
            print()

    def __iter__(self):
        self.__index = 0
        return self

    def __next__(self):
        if self.__index >= len(self.model_list):
            raise StopIteration
        self.__index+=1
        return self.model_list[self.__index-1]

    def __getitem__(self, key):
        return self.model_list[key]


    def save(self, outfile):
        if os.path.isdir(outfile):
            outfile = os.path.join(outfile, 'results.xlsx')

        run_name_l = []
        method_l = []
        df_name_l = []
        struc_l = []
        acc_l = []
        f1_l = []
        precision_l = []
        recall_l = [] 
        for model in self.model_list:
            run_name, method, df_name, struc, accuracy, f1, precision, recall = model.get_results()
            run_name_l.append(run_name)
            method_l.append(method)
            df_name_l.append(df_name)
            struc_l.append(struc)
            acc_l.append(accuracy)
            f1_l.append(f1)
            precision_l.append(precision)
            recall_l.append(recall)

        res_dict = {"Name":run_name_l, "Method": method_l, "DataType": df_name_l, "Structure": struc_l, 
            "Accuracy": acc_l, "F1":f1_l, "Precision":precision_l, "Recall":recall_l}
        res_df = pd.DataFrame.from_dict(res_dict)
        res_df.to_excel(outfile, header = True, index = False)




    def split_results_specific(self, parameter, value):
        test_obj = model() # just to check if given parameter is in obj dict
        if parameter not in test_obj.__dict__:
            return None

        split_results = model_results()
        for mobj in self:
            if mobj.__dict__[parameter] == value:
                split_results.add_model(mobj)

        return split_results



    def split_results_attribute(self, parameter):
        test_obj = model() # just to check if given parameter is in obj dict
        if parameter not in test_obj.__dict__:
            return None

        values = []
        model_result_l = []
        for mobj in self:
            if mobj.__dict__[parameter] not in values:
                values.append(mobj.__dict__[parameter])
        
        for v in values:
           model_result_l.append(self.split_results_specific(parameter, v))


        return values, model_result_l

    def split_results_values(self, parameter, value_l: list):
        test_obj = model() # just to check if given parameter is in obj dict
        if parameter not in test_obj.__dict__:
            return None
        
            
        model_result_l = []
        for v in value_l:
           model_result_l.append(self.split_results_specific(parameter, v))


        return model_result_l
            
        
    '''
    def plot_top_models(self, number, save_path, log_file):
        for mobj in self.model_list[0:number]:
            try:
                plotting_methods.plot_feature_importance(mobj, number = 10, 
                    title=mobj.run_name, save_path=save_path, disp=False)
            except Exception as e:
                with open(log_file, 'a') as f:
                    f.write("{}: {}\n".format(mobj.run_name, str(e)))
    '''

    def get_model(self, modelID):
        for enum, mobj in enumerate(self.model_list):
            if mobj.model_id == modelID:
                return mobj

    def print_specific_model_results(self, modelID):
        for enum, mobj in enumerate(self.model_list):
            if mobj.model_id == modelID:
                mobj.print_results(self.__longest_phrase)
                mobj.print_feature_importance()
                break
    
    def top_per_feature(self, print_res = False):
        phrases = ['Deep_grey', 'Brain_stem', 'Left_wm', 'Right_wm', 'Left_gm', 
                    'Right_gm', 'Left_cerebellum', 'Right_cerebellum']
        model_objs = []
        for phrase in phrases:        
            for enum, model_obj in enumerate(self.model_list):
                if model_obj.structure == phrase:
                    if print_res:
                        res = model_obj.get_results()
                        run_name, method, df_name, struc, accuracy, f1, precision, recall = res
                        print("{},{},{},{},{},{},{},{}".format(run_name, method, df_name, 
                                                        struc, accuracy, f1, precision, recall))
                    model_objs.append(model_obj)
                    break
        return model_objs

    
    def top_per_method(self, print_res = False):
        methods = ['LogR', 'SVM', 'XGB', 'RF']
        model_objs = []
        for meth in methods:
            for enum, model_obj in enumerate(self.model_list):
                if model_obj.method.lower() == meth.lower():
                    if print_res:
                        res = model_obj.get_results()
                        run_name, method, df_name, struc, accuracy, f1, precision, recall = res
                        print("{},{},{},{},{},{},{},{}".format(run_name, method, df_name, 
                                                        struc, accuracy, f1, precision, recall))
                    model_objs.append(model_obj)
                    break
        return model_objs

    def toDF(self, number = None):
        if number is None:
            number = len(self.model_list)
        cols = self.model_list[0].toSeries().index.tolist()
        for name in reversed(['run_name', 'df_name', 'structure', 'method']):
            cols.remove(name)
            cols.insert(0, name)
        df = pd.DataFrame(columns = cols)
        for mobj in self.model_list[:number]:
            s = mobj.toSeries()
            df.loc[len(df)] = s
        return df
