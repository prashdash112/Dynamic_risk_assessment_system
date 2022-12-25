import pandas as pd
import pickle
import os
from sklearn.metrics import f1_score
import json

#################Function for model scoring
def f1_score_model(model, test_data):
    '''
    Function to compute ML model F1 score
    '''
    X_test = test_data[['lastmonth_activity','lastyear_activity','number_of_employees']]
    y_test = test_data['exited']
    prediction = model.predict(X_test)
    score = f1_score(y_test, prediction)
    return score
 
if __name__ == "__main__":
    #################Load config.json and get path variables
    with open('config.json','r') as f:
        config = json.load(f) 

    # Output data, test data and model paths 
    dataset_csv_path = os.path.join(config['output_folder_path']) 
    output_data_path = os.getcwd() + '/' + dataset_csv_path + '/finaldata.csv'

    test_data_path = os.path.join(config['test_data_path'])
    test_data_path = os.getcwd() + '/' + test_data_path + '/testdata.csv'

    model_path = os.path.join(config['output_model_path'])  
    model_path = os.getcwd() + '/' + model_path + '/trainedmodel.pkl'

    # Datasets
    test_data = pd.read_csv(test_data_path)
    model = pickle.load(open(model_path, 'rb'))

    # Calling the score_model function 
    score = f1_score_model(model=model, test_data=test_data)
    #Saving the score to file
    score_path = os.getcwd() + '/' + config['output_model_path']
    with open(score_path + '/latestscore.txt', 'w') as file:
        file.write(str(score))


