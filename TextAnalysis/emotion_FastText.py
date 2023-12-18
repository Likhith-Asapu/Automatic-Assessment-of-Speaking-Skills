from datasets import load_dataset
import fasttext
import fasttext.util
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.model_selection import RepeatedStratifiedKFold
import re
from TrainSVM import * 
from TrainANN import *

dataset = load_dataset("Ar4ikov/iemocap_audio_text")
ft = fasttext.load_model('cc.en.100.bin')

    
def train_svm_classifier_for_emotion():
   
    # X,y = shuffle(X,y, random_state=42)
    X = []
    y = []

    def create_training_data(example):
        labels_encoded = {'ang':0, 'hap':1, 'sad':2, 'neu':3, 'dis':4, 'fea':5, 'sur':6, 'fru':7, 'exc':8, 'oth':9}
        if example['emotion'] == 'neu' or example['emotion'] == 'hap' or example['emotion'] == 'sad' or example['emotion'] == 'ang':
            sentence = example['to_translate']
            sentence = re.sub(r"[,.;@#?!&$-]",'',sentence)
            X.append(ft.get_sentence_vector(sentence))
            y.append(labels_encoded[example['emotion']])
    dataset.map(
        create_training_data
    )

    Train_SVM(X,y,"IEMOCAP text")
    Train_ANN(X,y,"IEMOCAP text",4)

train_svm_classifier_for_emotion()