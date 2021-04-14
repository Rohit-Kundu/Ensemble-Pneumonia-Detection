import pandas as pd
import numpy as np
from sklearn.metrics import *
import matplotlib.pyplot as plt
import math,os,argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_directory', type=str, default = './', help='Directory where csv files are stored')
args = parser.parse_args()

def getfile(filename):
    root="./"
    file = root+filename+'.csv'
    df = pd.read_csv(file,header=None)
    df = np.asarray(df)
    
    labels=[]
    for i in range(316):
        labels.append(0)
    for i in range(854):
        labels.append(1)
    
    labels = np.asarray(labels)
    return df,labels

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
    print("\nBalanced Accuracy Score: ",balanced_accuracy_score(labels,predictions))

def get_scores(labels,*argv):
    #outputs matrix of shape (no. of arg, 4) of precision, recall, f1-score, Area Under Curve
    count = len(argv)
    metrics = np.zeros(shape=(4,count))
    for i,arg in enumerate(argv):
        preds = predicting(arg)
        acc = accuracy_score(labels,preds)
        pre = precision_score(labels,preds)
        rec = recall_score(labels,preds)
        f1 = f1_score(labels,preds)
        auc = roc_auc_score(labels,preds)
        metrics[:,i] = np.array([pre,rec,f1,auc])
    weights = get_weights(np.transpose(metrics))
    print("Weights: ",weights)
    ensemble_prob = 0
    for i,arg in enumerate(argv):
        ensemble_prob+=weights[i]*arg
    return ensemble_prob

def anger_func(theta):
    result = (math.cos(2*theta-2*math.sin(theta)))/math.pi
    return result

def get_weights(matrix):
    weights = []
    for i in range(matrix.shape[0]):
        m = matrix[i]
        w = 0
        for j in range(m.shape[0]):
            w+=anger_func(m[j])
        weights.append(w)
    return weights

root = args.data_directory

if root[-1]!='/':
    root += '/'
csv_list = os.listdir(root)

p1,labels = getfile(root+csv_list[0].split('.')[0])
p2,_ = getfile(root+csv_list[1].split('.')[0])
p3,_ = getfile(root+csv_list[2].split('.')[0])

ensemble_prob = get_scores(labels,p1,p2,p3)


preds = predicting(ensemble_prob)
correct = np.where(preds == labels)[0].shape[0]
total = labels.shape[0]

print("Accuracy = ",correct/total)
classes = ['Normal','Pneumonia']
metrics(labels,preds,classes)
