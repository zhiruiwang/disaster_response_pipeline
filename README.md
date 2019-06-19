# Disaster Response Pipeline Project

### Project Overview
In this project, I analyzed disaster data from Figure Eight to build a model for an API that classifies disaster messages.

The data set contains real messages that were sent during disaster events. I created a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

I also included a web app where an emergency worker can input a new message and get classification results in several categories. The web app displays visualizations of the data.

Below are a few screenshots of the web app.
![](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/September/5b967bef_disaster-response-project1/disaster-response-project1.png)
![](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/September/5b967cda_disaster-response-project2/disaster-response-project2.png)

### Project Components
There are three components in this project.

1. ETL Pipeline
In process_data.py, it has a data cleaning pipeline that:

Loads the messages and categories datasets
Merges the two datasets
Cleans the data
Stores it in a SQLite database

2. ML Pipeline
In train_classifier.py, it has a machine learning pipeline that:

Loads data from the SQLite database
Splits the dataset into training and test sets
Builds a text processing and machine learning pipeline
Trains and tunes a model using GridSearchCV
Outputs results on the test set
Exports the final model as a pickle file

3. Flask Web App
In the last step, I'll display my results in a Flask web app. It has three plotly visualization of EDA on the data, and a query UI that can classsify the message.

### Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
