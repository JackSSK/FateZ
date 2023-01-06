import pandas as pd
import numpy as np
import os

class Reader():

    def dict_multiply(self,mat,dict_df):
        for i in dict_df:
            multiply_df = np.multiply(mat,dict_df[i])
            dict_df[i] = multiply_df
        return dict_df

    def output_csv_dict_df(self,dict_df,output_path):
        for i in dict_df:
            df_use = dict_df[i]
            if input_path[-1] != '/'
                input_path = input_path + '/'
            df_use.to_csv(output_path+i+'.csv')

    def input_csv_dict(self,input_path):
        file_list = os.listdir(input_path)
        dict_df = {}
        if input_path[-1] != '/'
            input_path = input_path + '/'
        for i in file_list:
            sample_name = i.split('.csv')
            sample = pd.read_csv(input_path+i)
            dict_df[sample_name] = sample
        return dict_df
