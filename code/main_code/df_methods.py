import pandas as pd
from sklearn.preprocessing import StandardScaler

import general_methods


phrases = general_methods.return_phrases()

def balance_df(df):
    small = None
    low_val = None
    for val in [-1, 0, 1]:
        res = (df.iloc[-1,:] == val).sum()
        if small is None or res < small:
            small = res
            low_val = val
            
    small_df = df.loc[:,df.iloc[-1,:] == -1].copy()
    norm_df = df.loc[:,df.iloc[-1,:] == 0].copy()
    large_df = df.loc[:,df.iloc[-1,:] == 1].copy()

    small_rand = general_methods.generate_rand_list(small, small_df.shape[1])
    norm_rand = general_methods.generate_rand_list(small, norm_df.shape[1])
    large_rand = general_methods.generate_rand_list(small, large_df.shape[1])


    small_new = small_df.iloc[:,small_rand]
    norm_new = norm_df.iloc[:,norm_rand]
    large_new = large_df.iloc[:,large_rand]
    
    final_df = pd.concat([small_new, norm_new, large_new], axis = 1)
    return final_df


# Create scaled DF 
def scale_df(data):
    temp_df = data.iloc[:-17,:].T.copy()
    scaler = StandardScaler()
    scaled_temp = scaler.fit_transform(temp_df)

    scaled_df = pd.DataFrame(scaled_temp)
    scaled_df.columns = temp_df.columns
    scaled_df.index = temp_df.index

    res = pd.concat([scaled_df.T, data.copy().iloc[-17:,:]], axis = 0)
    return res
    

# seperates strucutres into individual dataframes
# returns two lists of df's from normal and scaled for each structure, 8 total
def setup_data(data, scaled_data):
    row_nums = []
    dfs = []
    sdfs = []

    # seperate df's into structure specific dfs
    for p in phrases:
        ind_list = []
        for enum, n in enumerate(data.index):
            if p in n:
                ind_list.append(enum)
        row_nums.append((p, ind_list))
        df = data.iloc[ind_list,:]
        sdf = scaled_data.iloc[ind_list,:]

        dfs.append(df)
        sdfs.append(sdf)
        
    return dfs, sdfs
        
    
    
# create balanced datasets for better training
# input is list of dfs and scaled dfs from setup_data
def balance_data(dfs, sdfs):

    bdfs = []
    sbdfs = []
    for df, bdf in zip(dfs, sdfs):
        bdfs.append(balance_df(df))
        sbdfs.append(balance_df(bdf))

    return bdfs, sbdfs


# Print the distributions of cases between -1, 0, 1 (small, normal, large) for 
# each set of dataframes. 
def print_df_stats(dfs, bdfs):
    for df_list, label in zip([dfs, bdfs], ['Normal', 'Balanced']):
        print(label + "\n")
        for d, phrase in zip(df_list, phrases):
            print("{} DF:".format(phrase.replace('_', ' ').title()))
            for i, s in zip([-1, 0, 1], ['Small', 'Normal', 'Large']):    
                res = (d.iloc[-1,:] == i).sum()
                print("\t{}: {}".format(s, res))
            print("\n")