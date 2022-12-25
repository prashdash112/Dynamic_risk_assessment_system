
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess

##################Function to get model predictions
def model_predictions(model, test_data):
    #read the deployed model and a test dataset, calculate predictions
    X_test = test_data[['lastmonth_activity','lastyear_activity','number_of_employees']]
    prediction = model.predict(X_test)
    return prediction

##################Function to get summary statistics
def dataframe_summary(df):
    arr = []
    #calculate summary statistics here
    #return value should be a list containing all summary statistics
    arr.append(df.mean())
    arr.append(df.median())
    arr.append(df.std())
    return(arr)

##################Function to get missing values
def missing_data(df):
    percentage_na_vals = (df.isna().sum()/df.count())*100
    return percentage_na_vals

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    time_matrix = []

    starttime1 = timeit.default_timer()
    os.system('python ingestion.py')
    timing1=timeit.default_timer() - starttime1
    time_matrix.append(timing1)

    starttime2 = timeit.default_timer()
    os.system('python training.py')
    timing2=timeit.default_timer() - starttime2
    time_matrix.append(timing2)
    return time_matrix

##################Function to check dependencies
def outdated_packages_list():
    # List of outdated packages
    outdated_packages = []
    outdated = subprocess.check_output(['pip', 'list','--outdated'])
    outdated_packages.append(outdated)
    return outdated_packages

if __name__ == '__main__':

    ##################Load config.json and get environment variables
    with open('config.json','r') as f:
        config = json.load(f) 
    ## Final data
    dataset_csv_path = os.path.join(config['output_folder_path'])  
    dataset_csv_path = os.getcwd() + '/' + dataset_csv_path + '/finaldata.csv'
    ## test data
    test_data_path = os.path.join(config['test_data_path'])
    test_data_path = os.getcwd() + '/' + test_data_path + '/testdata.csv'
    # Model file
    model_path = os.path.join(config['output_model_path'])  
    model_path = os.getcwd() + '/' + model_path + '/trainedmodel.pkl'
    # Datasets
    test_data = pd.read_csv(test_data_path)
    model = pickle.load(open(model_path, 'rb'))

    #calling function 1
    result = model_predictions(model=model, test_data=test_data)
    print('Prediction result:\n\n',result,'\n')
    
    # Calling function 2 
    df = pd.read_csv(dataset_csv_path)
    summary = dataframe_summary(df[['lastmonth_activity','lastyear_activity','number_of_employees','exited']])
    summary = np.asarray(summary)
    print('Summary statistic: Mean Median and Std deviation(row wise)\n\n', summary)
    
    # Calling function 3 
    perc_na_vals = missing_data(df)
    print('\n\n percentage of missing values in each column of dataset:\n\n', perc_na_vals)

    #Calling function 4 
    time_monitor = execution_time()
    print('\n\n The time required by ingestion.py and training.py scripts to run\n\n', time_monitor)

    #Calling function 5
    outdated_package_list = outdated_packages_list()
    print('\n\n The list of packages\n\n:',outdated_package_list )