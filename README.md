SwiftML - Accelerating ML Journeys
SwiftML is an all-in-one machine learning assistant that simplifies and automates various stages of the ML workflow. With a focus on ease-of-use, SwiftML offers an intuitive interface for data preprocessing, model selection, and unsupervised-to-supervised learning transformation. This app requires no coding and provides results in just a few clicks.

Features
1. Data Preprocessing:
Missing Values Handling: Auto-fill missing values using strategies like mean, median, mode, or interpolation.

Data Cleaning: Automatically clean numeric and string columns, including gender identification.

Feature Scaling: Scale your features using standard or MinMax scaling methods.

Train-Test Split: Automatically split your dataset into training and testing sets.

2. Data Ingestion:
Upload and ingest datasets in CSV, JSON, or Excel formats.

Automatically save uploaded data to CSV for future processing.

3. Model Selection:
Regression & Classification: Automatically select the best machine learning model for your dataset (using PyCaret's setup and comparison features).

Model Evaluation: Evaluate and compare models' performance, with visual insights.

4. Clustering (Unsupervised → Supervised):
Convert unsupervised data to supervised data using KMeans clustering.

Label clusters and allow further supervised learning tasks.

Installation
To use SwiftML locally, follow these steps:

Prerequisites
Python 3.x

Streamlit

Scikit-learn

PyCaret

Steps to Install
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/AutoML.git
cd AutoML
Create a virtual environment:

bash
Copy
Edit
python -m venv venv
Activate the virtual environment:

Windows:

bash
Copy
Edit
.\venv\Scripts\activate
macOS/Linux:

bash
Copy
Edit
source venv/bin/activate
Install the required dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Usage
Start the Streamlit app by running:

bash
Copy
Edit
streamlit run app.py
Open the app in your browser (usually at http://localhost:8501).

Choose a section from the sidebar:

Data Preprocessing: Upload datasets and preprocess them (handling missing values, cleaning, and scaling).

Data Ingestion: Upload and ingest your datasets.

Model Selection: Select the target variable and let the app recommend the best model (for both regression and classification).

Clustering: Upload an unsupervised dataset, choose the number of clusters, and convert it to supervised data using KMeans.

Example Workflow
Data Preprocessing:

Upload a CSV or Excel file.

Choose your preferred missing value strategy, scaling method, and test size.

Download the preprocessed training and testing datasets.

Model Selection:

Once data is ingested, select the task (Regression or Classification) and target column.

The app will automatically recommend and evaluate models.

Clustering:

Upload an unsupervised dataset.

Choose the number of clusters for the KMeans algorithm.

Convert the data and download the clustered dataset.

