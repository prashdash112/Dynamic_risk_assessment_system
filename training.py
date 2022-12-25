import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

def splitter(path):
    '''
    Function to split the dataset in train-test split
    '''
    data = pd.read_csv(path)
    data = data.drop(['corporation'], axis=1, inplace=True)
    X = data[['lastmonth_activity','lastyear_activity','number_of_employees']]
    y = data['exited']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
    return X_train,X_test,y_train,y_test

def train_model(X_train,y_train,model_path):
    
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    model.fit(X_train,y_train)
    model_path_ = os.getcwd() + '/' + model_path #+'/trainedmodel.pkl'

    return pickle.dump(model, open(model_path_ + '/trainedmodel.pkl', 'wb'))

if __name__ == "__main__":
    ###################Load config.json and get path variables###############
    with open('config.json','r') as f:
        config = json.load(f) 
    dataset_csv_path = os.path.join(config['output_folder_path']) 
    model_path = os.path.join(config['output_model_path']) 
    output_folder_path = config['output_folder_path']
    finaldf_path = os.getcwd() + '/' + output_folder_path + '/finaldata.csv'
    ###############Splitting the dataset###############
    X_train,X_test,y_train,y_test = splitter(finaldf_path)
    ###########Training and saving the model to a location##############
    train_model(X_train,y_train,model_path)