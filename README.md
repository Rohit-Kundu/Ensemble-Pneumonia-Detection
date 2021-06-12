# Ensemble-Pneumonia-Detection

Official implementation of our paper titled "Pneumonia Detection from Chest X-ray Images using a Novel Weighted Average Ensemble Model" under peer review in Nature- PlosOne.

Abstract: Pneumonia is a respiratory infection caused by bacteria or viruses and affects a large population of individuals especially in developing and under-developed nations where risk factors like higher levels of pollution, unhygienic living conditions and overcrowding are more commonplace, along with inadequate medical infrastructure. Pneumonia causes pleural effusion, a condition where the lung gets filled with fluids causing difficulty in breathing. Early diagnosis of pneumonia is crucial for ensuring curative treatment and increasing the survival rate in the population. Chest X-ray imaging is the most common method for the diagnosis of pneumonia. However, examining chest X-rays is a daunting task and prone to subjective variability. In this research, we develop a Computer-Aided Diagnosis (CAD) system for the automatic detection of pneumonia from chest X-ray images. For this, we employ deep transfer learning to cope with the scarcity in available data and design an ensemble of three Convolutional Neural Network (CNN) models: GoogLeNet, ResNet-18 and DenseNet-121. A weighted average ensemble technique has been adopted wherein, the weights assigned to the base learners is determined using a novel approach. Here the scores four standard evaluation metrics: precision, recall, f1-score, the area under the curve (AUC), are fused to form the weights vector, instead of setting these experimentally like most work in literature, which is prone to errors. The proposed approach is evaluated on two publicly available pneumonia X-ray datasets using the 5-fold cross-validation scheme, achieving results that outperformed state-of-the-art methods on the same and performed superior to the popular ensemble techniques adopted in the literature. Further to justify the generalization capability of the approach, we perform additional tests on the skin cancer classification domain using the HAM10000 dataset where the results achieved are comparable to state-of-the-art.

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
|   +-- googlenet_train.csv
|   +-- resnet18_train.csv
|   +-- densenet121_train.csv
+-- test_csv
|   +-- .
|   +-- googlenet_test.csv
|   +-- resnet18_test.csv
|   +-- densenet121_test.csv
+-- main.py
+-- probability_extraction
+-- train_labels.csv
+-- test_labels.csv
```

To extract the probabilities on the validation set using the different models run `probability_extraction.py` and save the files in a folder.

Next, to run the ensemble model on the base learners run the following:

`python main.py --root_train "train_csv/" --root_test "test_csv/" --train_labels "train_labels.csv" --test_labels "test_labels.csv"`
