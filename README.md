# Disaster Response Pipeline Project

### Table of Contents

#### 1. [Instruction](#instruction)
#### 2. [File Description](#files)
#### 3. [Dependency](#dependency)
#### 4. [Acknowledgements](#ack)

## Instruction<a name="instruction"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## 2. File Description<a name = "files"></a>
1. App - This folder contains the python script to run the webapp
2. data - This folder contains the data and ipynb files, it also has python script to clean the data. It contains the database as .db file
3. model - This folder contains the python script that creates the classifier model


## 3. Dependencies<a name = "dependency"></a>
1. Python 3+
2. Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
3. Natural Language Process Libraries: NLTK
4. SQLlite Database Libraqries: SQLalchemy
5. Model Loading and Saving Library: Pickle
6. Web App and Data Visualization: Flask, Plotly


## 4. Acknowledgement<a name = "ack"></a>
Thanks to UDACITY for giving me the opportunity to build a project and for all the help. With that - play around with the code as much as you like!
