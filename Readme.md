# A dynamic risk assessment system

In this project, we are aiming at solving a company-client attrition problem. We wrote a logistic regression classifier to predict and flag the clients who are at a higher risk of leaving the firm.

## Several steps of the project:
### Data ingestion
In this step, we've collected data from various directories and save it as a final data in ingesteddata folder to be used for modelling. Also, data is  checked for duplicate values. 
### Training, Scoring and Deploying
In this step, we focus of creating and deploying a ML model from end to end. Training script is used for training the model using the finaldata.csv file.
Scoring script is used to claculate the F1 socre to measure the model performance.
Deploying script puts the saved model in the production_deployment directory from where it can be used in production and will be able to cater the needs of Users.
### Diagnostics
In this step, we focus on diagnosis of our ML model and look deep into the operational side of project. 
We wrote functions for:
1) Model predictions  - To calculate the predicted results
2) Summary statistics - To identify mean, median, std deviation of data
3) Missing data - To find the %age of missing data in input dataset
4) Timing - To calculate time required by ingestion and training script
5) Dependencies - To identify outdated dependencies
### Model reporting and API setup
In this step, we calculated and saved a confusion matrix to see how well our model is performing. Also, we setup a flask api to keep our key stakeholders updated with the progress and results. 
The api have 4 methods:
1) Prediction
2) scoring 
3) Summarystats
4) Diagnostics
### Process Automation
In this step, we look for data and model drift and if we found any significant drift then we aim at retraining of ingestion and training pipeline which in return update the results of our API. We wrote a fullprocess.py script to stich all components of the project. 
We aim at using a cronjob file to automate the pipeline and schedule it at every 10 mins.
Command to run the fullprocess.py file:

```
python fullprocess.py
```
## ##################################################################################################