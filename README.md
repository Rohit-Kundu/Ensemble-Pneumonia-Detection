# Ensemble-Pneumonia-Detection

Official implementation of our paper titled "Pneumonia Detection from Chest X-ray Images using a Novel Weighted Average Ensemble Model" under peer review in Nature- PlosOne.

## Requirements

To install the dependencies, run the following using the command prompt:

`pip install -r requirements.txt`

## Running the code on the Pneumonia data
To extract the probabilities on the validation set using the different models run `probability_extraction.py` and save the files in a folder.

Next, to run the ensemble model on the base learners run the following:

`python main.py --data_directory "pneumonia_csv/"`
