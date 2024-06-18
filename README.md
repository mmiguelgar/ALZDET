# MRI Classification for Alzheimer's Detection

This project is a simple web application that uses Django to classify MRI images to detect potential Alzheimer's disease. You can upload an MRI image and an Excel file (.xlsx) with clinical data, choose a machine learning model, and get a prediction.

## Features

- Upload MRI images for classification
- Upload clinical data in Excel format (.xlsx)
- Choose from different pre-trained machine learning models
- Get predictions based on the uploaded data

## Requirements

- Python 3.11
- Django 5.0.6
- Other dependencies listed in requirements.txt

## How to Install

1. Clone the repository:

git clone https://github.com/mmiguelgar/ALZDET.git
cd ALZDET

2. Create a virtual environment:

python -m venv myenv
myenv\Scripts\activate

3. Install the required packages:
pip install -r requirements.txt

4. Apply the database migrations:
python manage.py migrate

5. Start the server:
python manage.py runserver

6. Open your web browser and go to http://127.0.0.1:8000/

# How to Use
1. Upload an MRI image:

On the home page, upload an MRI image in NIfTI format (.nii).

2. Upload clinical data:

Upload an Excel file (.xlsx) with the following columns:

ID    M/F    Hand    Age    Educ    SES    MMSE    eTIV    nWBV    ASF    Delay

3. Select a model:

Choose one of the pre-trained models:

CNN Model GFC
CNN Model Masked GFC
CNN Combined Model

4. Get a prediction:

Click the "Upload" button to get the prediction results.

# Models
The following pre-trained models are available:

cnn_model_gfc.pkl
cnn_model_masked_gfc.pkl
cnn_combined_model.pkl

# Troubleshooting
If you have any issues:

- Make sure all dependencies are installed correctly
- Check that the file paths for the models are correct
- Ensure the uploaded files are in the correct format

# Contributing
We welcome contributions! Please fork the repository and submit pull requests.
