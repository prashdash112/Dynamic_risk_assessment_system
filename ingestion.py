import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

#############Function for data ingestion

def merge_multiple_dataframe():
    
    filenames = os.listdir(input_folder_path)
    ## input folder path
    dataset_csv_path = os.path.join(config['input_folder_path'])  
    dataset_csv_path = os.getcwd() + '/' + dataset_csv_path

    directory = dataset_csv_path + '/'
    # dummy dataframe
    final_df = pd.DataFrame(
        columns=[
            'corporation',
            'lastmonth_activity', 
            'lastyear_activity', 
            'number_of_employees', 
            'exited'
            ]
            )

    for file in filenames:
        path =  directory + file
        curr_df = pd.read_csv(path)
        final_df = final_df.append(curr_df).reset_index(drop=True)
    final_df = final_df.drop_duplicates()
    #saving the final dataset to the output folder path
    return final_df.to_csv(os.getcwd() + '/' + output_folder_path + '/finaldata.csv')

def drop_duplicates(df_path, output_file_path):
    df = pd.read_csv(df_path)
    df = df.drop_duplicates()
    return df.to_csv(output_file_path)

def ingestedfiles():
    '''
    Function to add input file details to ingestedfiles text file
    '''
    ingested_files = os.listdir(input_folder_path)
    with open(os.getcwd() + '/' + output_folder_path + '/ingestedfiles.txt', 'w') as file:
        for files in ingested_files:
            file.write(files+'\n')

if __name__ == '__main__':
    merge_multiple_dataframe()
    ingestedfiles()
