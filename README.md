# Ensemble-Pneumonia-Detection

Official implementation of our paper titled "Pneumonia Detection from Chest X-ray Images using a Novel Weighted Average Ensemble Model" under peer review in Nature- PlosOne.

## Requirements

To install the dependencies, run the following using the command prompt:

`pip install -r requirements.txt`

## Running the codes on Pneumonia data

The two datasets used in the paper can be found:
1. Dataset by Kermany et al.: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
2. RSNA Pneumonia Detection challenge dataset: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge

Required Directory Structure:
```
+-- data
|   +-- .
|   +-- train
|   +-- val
+-- train_csv
|   +-- .
|   +-- densenet121_train.csv
|   +-- googlenet_train.csv
|   +-- resnet18_train.csv
+-- test_csv
|   +-- .
|   +-- densenet121_test.csv
|   +-- googlenet_test.csv
|   +-- resnet18_test.csv
+-- main.py
+-- probability_extraction
+-- train_labels.csv
+-- test_labels.csv
```

To extract the probabilities on the validation set using the different models run `probability_extraction.py` and save the files in a folder.

Next, to run the ensemble model on the base learners run the following:

`python main.py --root_train "train_csv/" --root_test "test_csv/" --train_labels "train_labels.csv" --test_labels "test_labels.csv"`
