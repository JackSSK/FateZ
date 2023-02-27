import pandas as pd
import numpy as np
import os


def output_csv_dict_df(matrix_dict, output_path,sample=None):
    for i in matrix_dict:
        df_use = matrix_dict[i]
        if output_path[-1] != '/':
            output_path = output_path + '/'
        if sample == None:
            df_use.to_csv(output_path+i+'.csv')
        else:
            df_use.to_csv(output_path +sample+'#'+ i + '.csv')



def input_csv_dict_df(input_path):
    file_list = os.listdir(input_path)
    dict_df = {}
    if input_path[-1] != '/':
        input_path = input_path + '/'
    for i in file_list:
        sample_name = i.split('.csv')[0]
        sample = pd.read_csv(input_path+i,header=0,index_col=0)
        dict_df[sample_name] = sample
    return dict_df
