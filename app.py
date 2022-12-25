from flask import Flask, request
import pandas as pd
import pickle
from diagnostics import model_predictions, dataframe_summary, execution_time, outdated_packages_list
from diagnostics import missing_data, outdated_packages_list
from scoring import f1_score_model
import json
import os

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

## Training data path
dataset_csv_path = os.path.join(config['output_folder_path']) 
dataset_csv_path = os.getcwd() + '/' + dataset_csv_path + '/finaldata.csv'
## Model path 
model_path = os.path.join(config['output_model_path'])  
model_path = os.getcwd() + '/' + model_path + '/trainedmodel.pkl'
#model
model = pickle.load(open(model_path, 'rb'))


#######################Prediction Endpoint
@app.route("/prediction", methods=['GET','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    location = request.args.get('location')
    ## test data path
    test_data_path = os.getcwd() + '/' + location
    #Test data
    test_data = pd.read_csv(test_data_path)
    result = model_predictions(model=model, test_data=test_data)

    return 'Prediction results:\n\n' + str(result)

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():
    ## test data
    test_data_path = os.path.join(config['test_data_path'])
    test_data_path = os.getcwd() + '/' + test_data_path + '/testdata.csv'
    test_data = pd.read_csv(test_data_path)
    score = f1_score_model(model,test_data)
    return 'F1 Score of the model:\n\n' + str(score)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summarystat():        
    #check means, medians, and modes for each column
    df = pd.read_csv(dataset_csv_path)
    df = df[['lastmonth_activity','lastyear_activity','number_of_employees','exited']]
    summary = dataframe_summary(df)
    return 'Mean median and standard deviation of fields (Row wise)\n\n'+str(summary)

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostic():        
    #check timing and percent NA values and outdated dependencies
    df = pd.read_csv(dataset_csv_path)
    perc_na_vals = missing_data(df)
    ex_time = execution_time()
    outdated_package = outdated_packages_list()
    return 'Percentage NA vals in dataset:' + str(perc_na_vals) + '\n\n' + ' Execution time of ingestion and training script:' + str(ex_time) + '\n\n' + ' List of outdated packages:' + str(outdated_package) 

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
