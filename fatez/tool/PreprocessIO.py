import pandas as pd
import numpy as np
import os
from pkg_resources import resource_filename

def output_csv_dict_df(matrix_dict, output_path,sample=None):
    for i in matrix_dict:
        df_use = matrix_dict[i]
        if output_path[-1] != '/':
            output_path = output_path + '/'
        if sample == None:
            df_use.to_csv(output_path+i+'.csv')
        else:
            df_use.to_csv(output_path +sample+'#' + str(i)+'.csv')



def input_csv_dict_df(
        input_path,
        cell_use = None,
        order_cell = True,
        df_type = 'node',
        only_tf = False
        ):
    path = resource_filename(
        __name__, '../data/gene_order.txt'
    )
    gene_order = pd.read_table(path,header=None)
    if cell_use != None:
        file_list = cell_use
    else:
        file_list = os.listdir(input_path)
    dict_df = {}
    if input_path[-1] != '/':
        input_path = input_path + '/'
    for i in file_list:
        sample_name = i.split('.csv')[0]
        sample = pd.read_csv(input_path+i,header=0,index_col=0)
        if order_cell:
            if df_type == 'edge':
                sample = sample.reindex(columns=gene_order[0])
                sample = sample.reindex(gene_order[0][0:1103])
            elif df_type == 'node':
                sample = sample.reindex(gene_order[0])
        dict_df[sample_name] = sample
    return dict_df
