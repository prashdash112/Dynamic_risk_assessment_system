import os
import shutil
import json

def store_model_into_pickle(filename, src,target):
    '''
    Function to copy a file from 1 location to other location
    filename: File to copy
    src: Source destination
    target: Target destination
    '''
    shutil.copy(src+'/'+filename, target+'/'+filename)
    return 'Files coppied successfully'

if __name__ =="__main__":
    # Providing the folder path
    with open('config.json','r') as f:
        config = json.load(f) 

    model_path = os.path.join(config['output_model_path'])  
    origin1 = os.getcwd() + '/' + model_path

    ingested_path = os.path.join(config['output_folder_path']) 
    origin2 = os.getcwd() + '/' + ingested_path
    
    target_path =  os.path.join(config['prod_deployment_path'])
    target_path =   os.getcwd() + '/' + target_path
    
    store_model_into_pickle('latestscore.txt', origin1, target_path)
    store_model_into_pickle('trainedmodel.pkl', origin1, target_path)
    store_model_into_pickle('ingestedfiles.txt', origin2, target_path)