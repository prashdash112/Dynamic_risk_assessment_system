
import os
import json
import pickle
import pandas as pd
from scoring import f1_score_model
from training import *
from ingestion import *
from scoring import *
from diagnostics import *
from deployment import *
from reporting import *


#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f)

##################Check and read new data#########

prod_deployment_path =  os.path.join(config['prod_deployment_path'])
prod_deployment_path =   os.getcwd() + '/' + prod_deployment_path

with open(prod_deployment_path + '/' + 'ingestedfiles.txt', 'r') as file:
    existing_datasets = file.readlines()
existing_datasets = existing_datasets[0].split(' ') #dataset names in array format
print(existing_datasets)

## Input folder path
dataset_csv_path = os.path.join(config['input_folder_path'])  
dataset_csv_path = os.getcwd() + '/' + dataset_csv_path
new_datasets = os.listdir(dataset_csv_path)
print(new_datasets)

if existing_datasets != new_datasets:
    print("New Datasets found")
    os.system('python ingestion.py')
else:
    print('No new files found.Process finished.')
    
with open(prod_deployment_path+"/latestscore.txt", "r") as f:
    prev_f1 = float(f.read())

new_ingest_data = pd.read_csv(output_folder_path+"/finaldata.csv")
model = pickle.load(open(prod_deployment_path+"/trainedmodel.pkl", 'rb'))

new_f1 = f1_score_model(model, new_ingest_data)

if prev_f1 > new_f1:
    print(
        f'''Evidence of model drift 
        (prev_f1:{prev_f1:.4f}, new_f1:{new_f1:.4f}).
        Proceeding to re-deployment...'''
        )
    ##################Re-training
    os.system('python training.py')
    os.system('python scoring.py') 
    ##################Re-deployment
    #if you found evidence for model drift, re-run the deployment.py script
    os.system('python deployment.py')
    ##################Diagnostics and reporting
    #run diagnostics.py and reporting.py for the re-deployed model
    os.system('python apicalls.py')
    os.system('python reporting.py')

else:
    print(
        f'''There are no signs of model drift (prev_f1:{prev_f1:.4f}, new_f1:{new_f1:.4f}). 
        Process finished.'''
        )

#######################################END##########################################################






