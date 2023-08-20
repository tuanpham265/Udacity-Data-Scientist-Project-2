# Response Coordination for Disasters

Part of the Data Scientist Program by Udacity.

## Index
- Setting Up
- Why This Project?
- Directory Structure

## Setting Up:
1. **Data Processing:**
    - Clean and prepare your data with the ETL pipeline.
        ```bash
        python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
        ```
    - Use the Machine Learning pipeline to train and save a classifier.
        ```bash
        python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
        ```

2. **Web Application:**
    - From the `app` directory, initiate the web application with:
        ```bash
        python run.py
        ```
    - Access the application on: http://localhost:3001/

## Why This Project?
As a culminating project in the Advanced Data Analysis Program by Udacity, this project demonstrates the ability to handle real-world data. The main objective is to design a system that categorizes emergency messages, thereby facilitating the rapid and accurate dispatch to pertinent disaster relief organizations. The supplementary web interface, powered by Flask, allows on-ground personnel to instantly categorize messages during crisis management.

## Directory Structure
- **App Directory** (`app/`)
    - Main Application File (`run.py`)
    - Templates:
        - Classification Result (`go.html`)
        - Input Page (`master.html`)
- **Data Directory** (`data/`)
    - Data Processing Script (`process_data.py`)
    - Message Categories (`disaster_categories.csv`)
    - Disaster Messages (`disaster_messages.csv`)
    - Processed Database (`DisasterResponse.db`)
- **Model Directory** (`models/`)
    - Model Training Script (`train_classifier.py`)
    - Trained Classifier (`classifier.pkl`)
