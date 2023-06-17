# Disaster Response Pipeline Project

### Table of Contents

#### 1. [Introduction](#intro)
#### 1. [Instruction](#instruction)
#### 2. [File Description](#files)
#### 3. [Dependency](#dependency)
#### 4. [Acknowledgements](#ack)


## Introduction<a name="intro"></a>
This is a data science project which involves cleaning the data, getting it ready for building model and then getting the output to a webpage all using data pipelining and flask. In this project, I have build a model to classify messages sent during disasters. It has 36 message categories and we classify a message to be able to sent to their respective department and this provides a way to quickly help the people in need.


## Instruction<a name="instruction"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://192.168.1.10:3001/


## 2. File Description<a name = "files"></a>
~~~~~~~
 disaster_response_pipeline
          - app
                - templates
                        - go.html
                        - master.html
                - run.py
          - data
                - disaster_message.csv
                - disaster_categories.csv
                - ETL Pipeline Preparation.ipynb
                - ML Pipeline Preparation.ipynb
                - DisasterResponse.db
                - process_data.py
          - models
                - train_classifier.py
          - screenshots
                - best_parameters.png
                - categorisation.png
                - webapp.png
          - README.md
          - .gitignore
          - .DS_Store
~~~~~~~  

### Understanding Directories
1. App - This folder contains the python script to run the webapp
2. data - This folder contains the data and ipynb files, it also has python script to clean the data. It contains the database as .db file
3. model - This folder contains the python script that creates the classifier model
4. screenshots - Contains screenshots of the webapp and code logs


## 3. Dependencies<a name = "dependency"></a>
1. Python 3+
2. Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
3. Natural Language Process Libraries: NLTK
4. SQLlite Database Libraqries: SQLalchemy
5. Model Loading and Saving Library: Pickle
6. Web App and Data Visualization: Flask, Plotly


## 4. Acknowledgement<a name = "ack"></a>
Thanks to UDACITY for giving me the opportunity to build a project and for all the help. With that - play around with the code as much as you like!
