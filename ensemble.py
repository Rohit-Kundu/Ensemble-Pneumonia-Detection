import pandas as pd
import numpy as np
from sklearn.metrics import *
import matplotlib.pyplot as plt
import math,os,argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root_train', type=str, required = True, help='Directory where train csv files are stored')
parser.add_argument('--train_labels', type=str, required = True, help='File path for train labels')
parser.add_argument('--root_test', type=str, required = True, help='Directory where test csv files are stored')
parser.add_argument('--test_labels', type=str, required = True, help='File path for test labels')
args = parser.parse_args()


def getfile(filename):
    root="./"
    file = root+filename
    if '.csv' not in file:
        file+='.csv'
    df = pd.read_csv(file,header=None)
    df = np.asarray(df)[:,:-1] #Since last column has image names
    return df

def getlabels(filename):
    root="./"
    file = root+filename
    if '.csv' not in file:
        file+='.csv'
    df = pd.read_csv(file,header=None)
    df = np.asarray(df)[:,1] #Since first column has image names
    return df.astype(int)

def predicting(ensemble_prob):
    prediction = np.zeros((ensemble_prob.shape[0],))
    for i in range(ensemble_prob.shape[0]):
        temp = ensemble_prob[i]
        t = np.where(temp == np.max(temp))[0][0]
        prediction[i] = t
    return prediction

def metrics(labels,predictions,classes):
    print("Classification Report:")
    print(classification_report(labels, predictions, target_names = classes,digits = 4))
    matrix = confusion_matrix(labels, predictions)
    print("Confusion matrix:")
    print(matrix)
    print("\nClasswise Accuracy :{}".format(matrix.diagonal()/matrix.sum(axis = 1)))

def get_scores(labels,*argv):
    #outputs matrix of shape (no. of arg, 4) of precision, recall, f1-score, Area Under Curve
    count = len(argv)
    metrics = np.zeros(shape=(4,count))
    num_classes = np.unique(labels).shape[0]
    for i,arg in enumerate(argv):
        preds = predicting(arg)
        if num_classes==2:
            pre = precision_score(labels,preds)
            rec = recall_score(labels,preds)
            f1 = f1_score(labels,preds)
            auc = roc_auc_score(labels,preds)
        else:
            pre = precision_score(labels,preds,average='macro')
            rec = recall_score(labels,preds,average='macro')
            f1 = f1_score(labels,preds,average='macro')
            auc = roc_auc_score(labels,arg,average='macro',multi_class='ovo')
        metrics[:,i] = np.array([pre,rec,f1,auc])
    weights = get_weights(np.transpose(metrics))
    #print("Weights: ",weights)
    return weights

def get_weights(matrix):
    weights = []
    for i in range(matrix.shape[0]):
        m = matrix[i]
        w = 0
        for j in range(m.shape[0]):
            w+=np.tanh(m[j])
        weights.append(w)
    return weights

root_train = args.root_train
if root_train[-1]!='/':
    root_train += '/'

root_test = args.root_test
if root_test[-1]!='/':
    root_test += '/'

csv_list = os.listdir(root_train)
p1_train = getfile(root_train+csv_list[0])
p2_train = getfile(root_train+csv_list[1])
p3_train = getfile(root_train+csv_list[2])

train_labels = getlabels(args.train_labels)

p1_test = getfile(root_test+csv_list[0].replace('train','test'))
p2_test = getfile(root_test+csv_list[1].replace('train','test'))
p3_test = getfile(root_test+csv_list[2].replace('train','test'))

test_labels = getlabels(args.test_labels)

weights = get_scores(train_labels,p1_train,p2_train,p3_train)

ensemble_prob = weights[0]*p1_test+weights[1]*p2_test+weights[2]*p3_test
preds = predicting(ensemble_prob)
correct = np.where(preds == test_labels)[0].shape[0]
total = test_labels.shape[0]

print("Accuracy = ",correct/total)
classes = ['Normal','Pneumonia']
metrics(test_labels,preds,classes)
