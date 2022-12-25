import pickle
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt 
import json
import os
from diagnostics import model_predictions
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

##############Function for reporting
def score_model(y_true, y_pred):
    
    matrix = confusion_matrix(y_true,y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
    disp.plot()
    plt.savefig('confusionmatrix.png')

if __name__ == '__main__':
    
    ###############Load config.json and get path variables
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
    y_pred = model_predictions(model=model, test_data=test_data)
    y_true = test_data['exited'].values
    matrix = score_model(y_true, y_pred)
