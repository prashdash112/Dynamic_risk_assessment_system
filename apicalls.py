import requests
import json
import os
#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000/"

response1 = requests.post(url = URL + 'prediction', params={'location': 'testdata/testdata.csv'} )
response2 = requests.get(url = URL + 'scoring')
response3 = requests.get(url = URL + 'summarystats')
response4 = requests.get(url = URL + 'diagnostics')

responses = str(response1.content) + ' ' + str(response2.content) + ' ' + str(response3.content) + ' ' + str(response4.content) + ' '

with open('config.json','r') as f:
        config = json.load(f)

# Model file
model_path = os.path.join(config['output_model_path'])      
model_path = os.getcwd() + '/' + model_path

with open(model_path + '/'+'apireturns.txt', 'w') as file:
    file.write(responses)
