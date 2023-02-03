# A Dynamic Risk Assessment System

# Background
Imagine that you're the Chief Data Scientist at a big company that has 10,000 corporate clients. Your company is extremely concerned about attrition risk: the risk that some of their clients will exit their contracts and decrease the company's revenue. They have a team of client managers who stay in contact with clients and try to convince them not to exit their contracts. However, the client management team is small, and they're not able to stay in close contact with all 10,000 clients.

The company needs you to create, deploy, and monitor a risk assessment ML model that will estimate the attrition risk of each of the company's 10,000 clients. If the model you create and deploy is accurate, it will enable the client managers to contact the clients with the highest risk and avoid losing clients and revenue.

Creating and deploying the model isn't the end of your work, though. Your industry is dynamic and constantly changing, and a model that was created a year or a month ago might not still be accurate today. Because of this, you need to set up regular monitoring of your model to ensure that it remains accurate and up-to-date. You'll set up processes and scripts to re-train, re-deploy, monitor, and report on your ML model, so that your company can get risk assessments that are as accurate as possible and minimize client attrition.

# Project Steps Overview
We'll complete the project by proceeding through 5 steps:

1. Data ingestion. Automatically check a database for new data that can be used for model training. Compile all training data to a training dataset and save it to persistent storage. Write metrics related to the completed data ingestion tasks to persistent storage.
2. Training, scoring, and deploying. Write scripts that train an ML model that predicts attrition risk, and score the model. Write the model and the scoring metrics to persistent storage.
3. Diagnostics. Determine and save summary statistics related to a dataset. Time the performance of model training and scoring scripts. Check for dependency changes and package updates.
4. Reporting. Automatically generate plots and documents that report on model metrics. Provide an API endpoint that can return model predictions and metrics.
5. Process Automation. Create a script and cron job that automatically run all previous steps at regular intervals.

Using config.json Correctly
It's important to understand your config.json starter file since you'll be using it throughout the project. This file contains configuration settings for your project.

This file contains five entries:

* **input_folder_path,** which specifies the location where your project will look for input data, to ingest, and to use in model training. If you change the value of input_folder_path, your project will use a different directory as its data source, and that could change the outcome of every step.
* **output_folder_path**, which specifies the location to store output files related to data ingestion. In the starter version of config.json, this is equal to /ingesteddata/, which is the directory where you'll save your ingested data later in this step.
* **test_data_path**, which specifies the location of the test dataset
* **output_model_path**, which specifies the location to store the trained models and scores.
* **prod_deployment_path**, which specifies the location to store the models in production.


Seven locations should be aware of:

* **/practicedata/**. This is a directory that contains some data you can use for practice.

* **/sourcedata/**. This is a directory that contains data that you'll load to train your models.

* **/ingesteddata/**. This is a directory that will contain the compiled datasets after your ingestion script.

* **/testdata/**. This directory contains data you can use for testing your models.

* **/models/**. This is a directory that will contain ML models that you create for production.

* **/practicemodels/**. This is a directory that will contain ML models that you create as practice.

* **/production_deployment/**. This is a directory that will contain your final, deployed models.


**training.py**, a Python script meant to train an ML model

**scoring.py**, a Python script meant to score an ML model

**deployment.py**, a Python script meant to deploy a trained ML model

**ingestion.py**, a Python script meant to ingest new data

**diagnostics.py**, a Python script meant to measure model and data diagnostics

**reporting.py**, a Python script meant to generate reports about model metrics

**app.py**, a Python script meant to contain API endpoints

**wsgi.py**, a Python script to help with API deployment

**apicalls.py**, a Python script meant to call your API endpoints

**fullprocess.py**, a script meant to determine whether a model needs to be re-deployed, and to call all other Python scripts when needed

## Step 1: Data Ingestion

We'll read data files into Python, and write them to an output file that will be our master dataset. We'll also save a record of the files we've read.

## Step 2: Training, Scoring, and Deploying an ML Model

This step will require to write three scripts.
* training.py, a Python script that will accomplish model training

Read in finaldata.csv using the pandas module

Train the ML model using the specified algorithm and requirements

save the trained model to a file called trainedmodel.pkl

* scoring.py, a Python script that will accomplish model scoring

Take an ML model and a test dataset as inputs.

Calculate the F1 score of the model on the test dataset.

Write the F1 score to latestscore.txt.

* deployment.py, a Python script that will accomplish model deployment

Copies trainedmodel.pkl to the produciton deployment directory.

Copies latestscore.txt to the production deployment directory.

Copies ingestedfiles.txt to the production deployment directory.

## Step 3: Model and Data Diagnostics

In this step, we'll create a script diagnostics.py that performs diagnostic tests related to our model as well as our data.

Calculate summary statistics(mean, median, and standard deviation) for each numeric column in a dataset.

Calculate the percent of each column that consists of NA values.

Measure the timing of important ML tasks(data ingestion and training).

Check whether the module dependencies are up-to-data.

Make predictions for an input dataset using the current deployed model.

## Step 4: Model Reporting 
For this step, we're going to work with two scripts from our collection of starter files: reporting.py - which we'll use for generating plots, and app.py - which we'll use for API setup.

Generate aplot of a confusion matrix for any input data and model.

Enable an API endpoint that returns model predictions, model scores, model summary statistics, model diagnostics including timing measurements, missing data and dependency check.

## Step 5: Process Automation

In this step, we'll create scripts  fullprocess.py that automate the ML model scoring and monitoring process.

Check for new data, and read new files if new data exists.

Check for model drift.

Re-deploy if there is new data and if there is model drift.

Run apicalls.py and reporing.py on the latest deployed model.

Set up a cron job that automatically runs fullprocess.py once every 10 min.